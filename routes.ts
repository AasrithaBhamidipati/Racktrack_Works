import type { Express } from "express";
import express from "express";
import session from "express-session";
import createMemoryStore from "memorystore";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { pool } from "./db";
import { loginCredentialsSchema, insertUploadSchema, updateProfileSchema } from "@shared/schema";
import multer from "multer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { randomUUID } from "crypto";
import { exec } from "child_process";
import { promisify } from "util";
import { spawn } from "child_process";
import { startJobWorker, getWorkerStatus } from "./jobWorker";
import * as logger from "./logger";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const execAsync = promisify(exec);

const filesDir = "files";
if (!fs.existsSync(filesDir)) {
  fs.mkdirSync(filesDir, { recursive: true });
}

const jobsOutputDir = path.join(process.cwd(), "jobs_output");
if (!fs.existsSync(jobsOutputDir)) {
  fs.mkdirSync(jobsOutputDir, { recursive: true });
}

const uploadTypeFolders = ["single-image", "multiple-images", "video"];
uploadTypeFolders.forEach((folder) => {
  const folderPath = path.join(filesDir, folder);
  if (!fs.existsSync(folderPath)) {
    fs.mkdirSync(folderPath, { recursive: true });
  }
});

const allowedTypes: Record<
  string,
  { mimes: string[]; extensions: string[]; maxFiles: number }
> = {
  "single-image": {
    mimes: ["image/jpeg", "image/png", "image/jpg", "image/webp"],
    extensions: [".jpg", ".jpeg", ".png", ".webp"],
    maxFiles: 1,
  },
  "multiple-images": {
    mimes: ["image/jpeg", "image/png", "image/jpg", "image/webp"],
    extensions: [".jpg", ".jpeg", ".png", ".webp"],
    maxFiles: 20,
  },
  video: {
    mimes: ["video/mp4", "video/webm", "video/quicktime"],
    extensions: [".mp4", ".webm", ".mov"],
    maxFiles: 1,
  },
};

const tempDir = path.join(filesDir, "temp");
if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir, { recursive: true });
}

const uploadStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, tempDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const basename = path.basename(file.originalname, ext);
    const safeName = basename.replace(/[^a-zA-Z0-9-_]/g, "_");
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, `${uniqueSuffix}-${safeName}${ext}`);
  },
});

const upload = multer({
  storage: uploadStorage,
  limits: {
    fileSize: 500 * 1024 * 1024,
  },
});

declare module "express-session" {
  interface SessionData {
    userId?: string;
    username?: string;
    currentJobId?: string;
  }
}

export async function registerRoutes(app: Express): Promise<Server> {
  const MemoryStore = createMemoryStore(session);

  const sessionStore = new MemoryStore({
    checkPeriod: 86400000,
  });

  app.use(
    session({
      secret: process.env.SESSION_SECRET || "racktrack-secret-key-change-in-production",
      resave: false,
      saveUninitialized: false,
      store: sessionStore,
      cookie: {
        maxAge: 86400000,
        httpOnly: true,
        secure: false,
      },
    })
  );

  startJobWorker();

  app.use(express.json());
  app.use(express.urlencoded({ extended: false }));

  app.post("/api/upload", upload.array("files", 20), async (req, res) => {
    try {
      const uploadType = req.body.uploadType;
      const files = req.files as Express.Multer.File[];

      if (!uploadType || !allowedTypes[uploadType]) {
        if (req.files) {
          (req.files as Express.Multer.File[]).forEach((file) => {
            if (fs.existsSync(file.path)) {
              fs.unlinkSync(file.path);
            }
          });
        }
        return res
          .status(400)
          .json({ success: false, message: "Invalid upload type" });
      }

      if (!files || files.length === 0) {
        return res.status(400).json({
          success: false,
          message: "No files uploaded or invalid file types",
        });
      }

      const typeConfig = allowedTypes[uploadType];

      const invalidFiles = files.filter((file) => {
        const ext = path.extname(file.originalname).toLowerCase();
        const isMimeAllowed = typeConfig.mimes.includes(file.mimetype);
        const isExtAllowed = typeConfig.extensions.includes(ext);
        return !isMimeAllowed || !isExtAllowed;
      });

      if (invalidFiles.length > 0) {
        files.forEach((file) => fs.unlinkSync(file.path));
        return res.status(400).json({
          success: false,
          message:
            "Invalid file type(s). Please upload only allowed file formats.",
        });
      }

      if (files.length > typeConfig.maxFiles) {
        files.forEach((file) => fs.unlinkSync(file.path));
        return res.status(400).json({
          success: false,
          message: `Too many files. Maximum ${typeConfig.maxFiles} allowed for ${uploadType}`,
        });
      }

      const userId = req.session?.userId || null;
      const createdJobs: any[] = [];

      const jobId = `job_${Date.now()}_${randomUUID().slice(0, 8)}`;
      const jobInputDir = path.join(jobsOutputDir, jobId, "input");
      const jobOutputDir = path.join(jobsOutputDir, jobId, "output");

      fs.mkdirSync(jobInputDir, { recursive: true });
      fs.mkdirSync(jobOutputDir, { recursive: true });

      const uploadedFiles = await Promise.all(
        files.map(async (file) => {
          const fileName = path.basename(file.path);
          const newPath = path.join(jobInputDir, fileName);

          fs.renameSync(file.path, newPath);

          const uploadData = {
            fileName: file.originalname,
            fileType: file.mimetype,
            filePath: newPath,
            uploadType: uploadType,
          };

          const validatedData = insertUploadSchema.parse(uploadData);
          return await storage.createUpload(validatedData);
        }),
      );

      let inputPath: string;
      if (uploadType === "multiple-images") {
        inputPath = jobInputDir;
      } else {
        inputPath = uploadedFiles[0].filePath;
      }

      const firstUploadId = uploadedFiles[0]?.id || null;

      const job = await storage.createJob({
        jobId,
        userId: userId,
        uploadId: firstUploadId,
        jobType: "cpu",
        status: "waiting",
        inputPath: inputPath,
        outputPath: jobOutputDir,
      });

      await storage.addJobLog(jobId, `Job created for ${uploadedFiles.length} file(s)`);

      req.session.currentJobId = jobId;

      createdJobs.push({
        jobId: job.jobId,
        status: job.status,
        outputPath: jobOutputDir,
      });

      res.json({
        success: true,
        message: `Successfully uploaded ${uploadedFiles.length} file(s). Job queued for processing.`,
        uploads: uploadedFiles,
        jobs: createdJobs,
      });
    } catch (error) {
      console.error("Upload error:", error);
      if (req.files) {
        (req.files as Express.Multer.File[]).forEach((file) => {
          if (fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
          }
        });
      }
      res.status(500).json({ success: false, message: "Upload failed" });
    }
  });

  app.post("/api/register", async (req, res) => {
    try {
      const { username, password } = req.body as { username?: string; password?: string };
      if (!username || !password) {
        return res.status(400).json({ success: false, message: "username and password are required" });
      }

      const existing = await storage.getUserByUsername(username);
      if (existing) {
        return res.status(400).json({ success: false, message: "Username already exists" });
      }

      const user = await storage.createUser({ username, password });
      res.status(201).json({ success: true, message: "User registered", user: { id: user.id, username: user.username } });
    } catch (err) {
      console.error("Register error:", err);
      res.status(500).json({ success: false, message: "Registration failed" });
    }
  });

  app.post("/api/login", async (req, res) => {
    try {
      const start = Date.now();
      const credentials = loginCredentialsSchema.parse(req.body);
      logger.info(`Login attempt for ${credentials.username}`, "auth");

      const validateStart = Date.now();
      const isValid = await storage.validateCredentials(credentials);
      logger.info(`validateCredentials took ${Date.now() - validateStart}ms`, "auth");

      if (!isValid) {
        logger.info(`Login failed for ${credentials.username} (invalid) after ${Date.now() - start}ms`, "auth");
        return res.status(401).json({ success: false, message: "Invalid credentials" });
      }

      const userFetchStart = Date.now();
      const user = await storage.getUserByUsername(credentials.username);
      logger.info(`getUserByUsername took ${Date.now() - userFetchStart}ms`, "auth");

      if (req.session) {
        req.session.userId = user?.id;
        req.session.username = credentials.username;
      }

      logger.info(`Login successful for ${credentials.username} in ${Date.now() - start}ms`, "auth");
      res.json({ 
        success: true, 
        message: "Login successful", 
        user: user ? { id: user.id, username: user.username } : undefined 
      });
    } catch (error) {
      logger.error("Login error", error, "auth");
      res.status(400).json({ success: false, message: "Invalid request" });
    }
  });

  app.post("/api/logout", (req, res) => {
    if (req.session) {
      req.session.destroy(() => {});
    }
    res.json({ success: true, message: "Logged out" });
  });

  app.get("/api/session", (req, res) => {
    if (req.session?.userId) {
      return res.json({ success: true, user: { id: req.session.userId, username: req.session.username } });
    }
    return res.status(401).json({ success: false, message: "Not authenticated" });
  });

  app.get("/api/user/profile", async (req, res) => {
    try {
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ success: false, message: "User not found" });
      }

      const { password, ...userWithoutPassword } = user;
      res.json({ success: true, user: userWithoutPassword });
    } catch (error) {
      console.error("Get profile error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch profile" });
    }
  });

  app.patch("/api/user/profile", async (req, res) => {
    try {
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const profileData = updateProfileSchema.parse(req.body);
      const updatedUser = await storage.updateUserProfile(userId, { ...profileData, profileImage: req.body.profileImage });

      if (!updatedUser) {
        return res.status(404).json({ success: false, message: "User not found" });
      }

      const { password, ...userWithoutPassword } = updatedUser;
      res.json({ success: true, message: "Profile updated", user: userWithoutPassword });
    } catch (error) {
      console.error("Update profile error:", error);
      res.status(400).json({ success: false, message: "Failed to update profile" });
    }
  });

  app.get("/api/jobs", async (req, res) => {
    try {
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const jobs = await storage.getUserJobs(userId);
      res.json({ success: true, jobs });
    } catch (error) {
      console.error("Get jobs error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch jobs" });
    }
  });

  app.get("/api/jobs/:jobId", async (req, res) => {
    try {
      const { jobId } = req.params;
      const job = await storage.getJob(jobId);

      if (!job) {
        return res.status(404).json({ success: false, message: "Job not found" });
      }

      const logs = await storage.getJobLogs(jobId);

      res.json({ success: true, job, logs });
    } catch (error) {
      console.error("Get job error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch job" });
    }
  });

  app.get("/api/jobs/:jobId/output", async (req, res) => {
    try {
      const { jobId } = req.params;
      const job = await storage.getJob(jobId);

      if (!job) {
        return res.status(404).json({ success: false, message: "Job not found" });
      }

      if (!job.outputPath || !fs.existsSync(job.outputPath)) {
        return res.status(404).json({ success: false, message: "Output not available yet" });
      }

      const files: { path: string; name: string; type: string }[] = [];

      const walkDir = (dir: string, basePath: string = ""): void => {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          const relativePath = path.join(basePath, entry.name);

          if (entry.isDirectory()) {
            walkDir(fullPath, relativePath);
          } else {
            files.push({
              path: relativePath,
              name: entry.name,
              type: path.extname(entry.name).slice(1),
            });
          }
        }
      };

      walkDir(job.outputPath);

      res.json({ success: true, jobId, outputPath: job.outputPath, files });
    } catch (error) {
      console.error("Get job output error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch job output" });
    }
  });

  app.get("/api/worker/status", (req, res) => {
    const status = getWorkerStatus();
    res.json({ success: true, ...status });
  });

  app.get("/api/uploads", async (req, res) => {
    try {
      const uploads = await storage.getAllUploads();
      res.json({ success: true, uploads });
    } catch (error) {
      res.status(500).json({ success: false, message: "Failed to fetch uploads" });
    }
  });

  app.get("/api/segments/image", async (req, res) => {
    try {
      const imagePath = req.query.path as string;

      if (!imagePath) {
        return res.status(400).json({ success: false, message: "No image path provided" });
      }

      const resolvedPath = path.join(process.cwd(), imagePath);
      const segmentedOutputDir = path.join(process.cwd(), "segmented_output");
      const uploadsDir = path.join(process.cwd(), "uploads");
      const filesDir = path.join(process.cwd(), "files");
      const jobsDir = path.join(process.cwd(), "jobs_output");

      const isInSegmentedOutput = resolvedPath.startsWith(segmentedOutputDir);
      const isInUploads = resolvedPath.startsWith(uploadsDir);
      const isInFiles = resolvedPath.startsWith(filesDir);
      const isInJobs = resolvedPath.startsWith(jobsDir);

      if (!isInSegmentedOutput && !isInUploads && !isInFiles && !isInJobs) {
        return res.status(403).json({ success: false, message: "Access denied" });
      }

      if (!fs.existsSync(resolvedPath)) {
        return res.status(404).json({ success: false, message: "Image not found" });
      }

      res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, private");
      res.setHeader("Pragma", "no-cache");
      res.setHeader("Expires", "0");

      res.sendFile(resolvedPath);
    } catch (error) {
      console.error("Error serving image:", error);
      res.status(500).json({ success: false, message: "Failed to serve image" });
    }
  });

  app.get("/api/segments/:folder", async (req, res) => {
    try {
      const folder = req.params.folder;
      const currentJobId = req.session.currentJobId;

      console.log(`[/api/segments] folder: ${folder}, currentJobId: ${currentJobId}`);

      if (!currentJobId) {
        console.log("[/api/segments] No currentJobId in session");
        return res.json([]);
      }

      let job = await storage.getJob(currentJobId);
      console.log(`[/api/segments] job found in db: ${!!job}, outputPath: ${job?.outputPath}`);
      
      // If job not in database, try to find it in the filesystem
      if (!job) {
        console.log(`[/api/segments] Job not in database, checking filesystem for: ${currentJobId}`);
        try {
          const jobPath = path.join(process.cwd(), "jobs_output", currentJobId, "output");
          if (fs.existsSync(jobPath)) {
            job = {
              jobId: currentJobId,
              userId: req.session.userId || "unknown",
              outputPath: jobPath,
              status: "completed",
              fromFilesystem: true
            };
            console.log(`[/api/segments] Found job in filesystem: ${jobPath}`);
          }
        } catch (fsError) {
          console.error(`[/api/segments] Error checking filesystem for job:`, fsError);
        }
      }
      
      if (!job || !job.outputPath) {
        console.log("[/api/segments] Job not found or no outputPath");
        return res.json([]);
      }

      // Python script outputs class folders directly to outputPath
      const outputDir = job.outputPath;

      if (!fs.existsSync(outputDir)) {
        console.log(`[/api/segments] outputDir does not exist: ${outputDir}`);
        return res.json([]);
      }

      console.log(`[/api/segments] outputDir exists: ${outputDir}`);
      const images: { path: string; name: string }[] = [];
      const imageExts = [".jpg", ".jpeg", ".png", ".webp"];

      // Special handling for "switch" - extract from units folder structure
      if (folder === "switch") {
        const unitsPath = path.join(outputDir, "units");
        if (fs.existsSync(unitsPath)) {
          const units = fs.readdirSync(unitsPath, { withFileTypes: true }).sort();
          for (const unit of units) {
            if (unit.isDirectory()) {
              const switchPath = path.join(unitsPath, unit.name, "switch");
              if (fs.existsSync(switchPath)) {
                const files = fs.readdirSync(switchPath).sort();
                files.forEach((file) => {
                  const ext = path.extname(file).toLowerCase();
                  if (imageExts.includes(ext)) {
                    const relativePath = path.join("jobs_output", currentJobId, "output", "units", unit.name, "switch", file).replace(/\\/g, "/");
                    images.push({
                      path: relativePath,
                      name: `${unit.name} - ${file}`,
                    });
                  }
                });
              }
            }
          }
        }
        return res.json(images);
      }

      // Special handling for "patch_panel" - extract from units folder structure
      if (folder === "patch_panel" || folder === "patch-panel") {
        const unitsPath = path.join(outputDir, "units");
        if (fs.existsSync(unitsPath)) {
          const units = fs.readdirSync(unitsPath, { withFileTypes: true }).sort();
          for (const unit of units) {
            if (unit.isDirectory()) {
              const patchPanelPath = path.join(unitsPath, unit.name, "patchpanel");
              if (fs.existsSync(patchPanelPath)) {
                const files = fs.readdirSync(patchPanelPath).sort();
                files.forEach((file) => {
                  const ext = path.extname(file).toLowerCase();
                  if (imageExts.includes(ext)) {
                    const relativePath = path.join("jobs_output", currentJobId, "output", "units", unit.name, "patchpanel", file).replace(/\\/g, "/");
                    images.push({
                      path: relativePath,
                      name: `${unit.name} - ${file}`,
                    });
                  }
                });
              }
            }
          }
        }
        return res.json(images);
      }

      // Special handling for "connected_port" - extract from units/*/ports/connected/
      if (folder === "connected_port" || folder === "connected-port") {
        const unitsPath = path.join(outputDir, "units");
        if (fs.existsSync(unitsPath)) {
          const units = fs.readdirSync(unitsPath, { withFileTypes: true }).sort();
          for (const unit of units) {
            if (unit.isDirectory()) {
              const connectedPortPath = path.join(unitsPath, unit.name, "ports", "connected");
              if (fs.existsSync(connectedPortPath)) {
                const files = fs.readdirSync(connectedPortPath).sort();
                files.forEach((file) => {
                  const ext = path.extname(file).toLowerCase();
                  if (imageExts.includes(ext)) {
                    const relativePath = path.join("jobs_output", currentJobId, "output", "units", unit.name, "ports", "connected", file).replace(/\\/g, "/");
                    images.push({
                      path: relativePath,
                      name: `${unit.name} - Connected - ${file}`,
                    });
                  }
                });
              }
            }
          }
        }
        return res.json(images);
      }

      // Special handling for "empty_port" - extract from units/*/ports/empty/
      if (folder === "empty_port" || folder === "empty-port") {
        const unitsPath = path.join(outputDir, "units");
        if (fs.existsSync(unitsPath)) {
          const units = fs.readdirSync(unitsPath, { withFileTypes: true }).sort();
          for (const unit of units) {
            if (unit.isDirectory()) {
              const emptyPortPath = path.join(unitsPath, unit.name, "ports", "empty");
              if (fs.existsSync(emptyPortPath)) {
                const files = fs.readdirSync(emptyPortPath).sort();
                files.forEach((file) => {
                  const ext = path.extname(file).toLowerCase();
                  if (imageExts.includes(ext)) {
                    const relativePath = path.join("jobs_output", currentJobId, "output", "units", unit.name, "ports", "empty", file).replace(/\\/g, "/");
                    images.push({
                      path: relativePath,
                      name: `${unit.name} - Empty - ${file}`,
                    });
                  }
                });
              }
            }
          }
        }
        return res.json(images);
      }

      // Fallback: combined "ports" handler - extract from units/*/ports/{connected,empty}/
      if (folder === "ports") {
        const unitsPath = path.join(outputDir, "units");
        if (fs.existsSync(unitsPath)) {
          const units = fs.readdirSync(unitsPath, { withFileTypes: true }).sort();
          for (const unit of units) {
            if (unit.isDirectory()) {
              const portsPath = path.join(unitsPath, unit.name, "ports");
              if (fs.existsSync(portsPath)) {
                // Check for connected and empty port subfolders
                const portTypes = ["connected", "empty"];
                for (const portType of portTypes) {
                  const portTypePath = path.join(portsPath, portType);
                  if (fs.existsSync(portTypePath)) {
                    const files = fs.readdirSync(portTypePath).sort();
                    files.forEach((file) => {
                      const ext = path.extname(file).toLowerCase();
                      if (imageExts.includes(ext)) {
                        const relativePath = path.join("jobs_output", currentJobId, "output", "units", unit.name, "ports", portType, file).replace(/\\/g, "/");
                        const portTypeLabel = portType === "connected" ? "Connected" : "Empty";
                        images.push({
                          path: relativePath,
                          name: `${unit.name} - ${portTypeLabel} - ${file}`,
                        });
                      }
                    });
                  }
                }
              }
            }
          }
        }
        return res.json(images);
      }

      // Map expected folder names to actual folder names in output
      const folderMapping: Record<string, string[]> = {
        "cables": ["cables", "cable"],
        "rack": ["rack"],
        "patch_panel": ["patch_panel", "patch-panel"],
        "switch": ["switch"],
        "connected_port": ["connected_port", "connected-port"],
        "empty_port": ["empty_port", "empty-port"],
        "units": ["units"]
      };

      const possibleFolders = folderMapping[folder] || [folder];
      
      let actualFolder: string | null = null;
      for (const possibleName of possibleFolders) {
        const testPath = path.join(outputDir, possibleName);
        if (fs.existsSync(testPath) && fs.statSync(testPath).isDirectory()) {
          actualFolder = possibleName;
          break;
        }
      }

      if (actualFolder) {
        const folderPath = path.join(outputDir, actualFolder);
        const files = fs.readdirSync(folderPath);
        files.forEach((file) => {
          const ext = path.extname(file).toLowerCase();
          if (imageExts.includes(ext)) {
            const relativePath = path.join("jobs_output", currentJobId, "output", actualFolder!, file).replace(/\\/g, "/");
            images.push({
              path: relativePath,
              name: file,
            });
          }
        });
      }

      res.json(images);
    } catch (error) {
      console.error("Error fetching segments:", error);
      res.status(500).json({ success: false, message: "Failed to fetch segments" });
    }
  });

  // Helper to run report generation using report.py, then 8_json.py
  async function runReportForJob(job: any) {
    const jobResultsDir = path.join(job.outputPath, "Results");
    const segmentedOutputDir = job.outputPath;

    const executionLogs: Array<{
      script: string;
      status: "success" | "error";
      stdout: string;
      stderr: string;
      error?: string;
      timestamp: string;
    }> = [];

    try {
      // Ensure Results directory exists
      if (!fs.existsSync(jobResultsDir)) {
        fs.mkdirSync(jobResultsDir, { recursive: true });
      }

      await storage.addJobLog(job.jobId, "Report generation started using report.py");

      // ============ STEP 1: Run report.py ============
      const reportScriptPath = path.join(process.cwd(), "python_codes", "report.py");
      const pythonExe = "python";
      const reportArgs = [
        reportScriptPath,
        "--input", segmentedOutputDir,
        "--output", jobResultsDir
      ];

      let timestamp = new Date().toISOString();
      await storage.addJobLog(job.jobId, `Running: report.py --input ${segmentedOutputDir} --output ${jobResultsDir}`);
      console.log(`[Report Generation] Spawning: report.py`);

      let reportOut = "";
      let reportErr = "";
      
      const reportExitCode: number = await new Promise((resolve, reject) => {
        const child = spawn(pythonExe, reportArgs, { cwd: process.cwd() });

        child.stdout.on("data", (chunk) => {
          const s = chunk.toString();
          reportOut += s;
          storage.addJobLog(job.jobId, s).catch(() => {});
          console.log(`[report.py]`, s);
        });

        child.stderr.on("data", (chunk) => {
          const s = chunk.toString();
          reportErr += s;
          storage.addJobLog(job.jobId, s).catch(() => {});
          console.error(`[report.py Error]`, s);
        });

        child.on("error", (e) => reject(e));
        child.on("close", (code) => resolve(code === null ? 1 : code));
      });

      if (reportExitCode !== 0) {
        executionLogs.push({ 
          script: "report.py", 
          status: "error", 
          stdout: reportOut, 
          stderr: reportErr, 
          error: `Exit code ${reportExitCode}`, 
          timestamp 
        });
        await storage.addJobLog(job.jobId, `report.py failed with exit code ${reportExitCode}`);
        await storage.updateJobStatus(job.jobId, "failed", "Report generation failed");
        return executionLogs;
      }

      executionLogs.push({ 
        script: "report.py", 
        status: "success", 
        stdout: reportOut, 
        stderr: reportErr, 
        timestamp 
      });
      await storage.addJobLog(job.jobId, "report.py completed successfully");

      // Clean up unnecessary files from Results directory
      try {
        const filesToDelete = [
          path.join(jobResultsDir, "Refers_images"),
          path.join(jobResultsDir, "Results"),
          path.join(jobResultsDir, "_merge_tmp")
        ];

        for (const filePath of filesToDelete) {
          if (fs.existsSync(filePath)) {
            const stats = fs.statSync(filePath);
            if (stats.isDirectory()) {
              fs.rmSync(filePath, { recursive: true, force: true });
              await storage.addJobLog(job.jobId, `Cleaned up: ${path.basename(filePath)}`);
            }
          }
        }
      } catch (cleanupErr) {
        console.warn("Cleanup warning:", cleanupErr);
        // Continue even if cleanup fails
      }

      // ============ STEP 2: Run 8_json.py ============
      const jsonScriptPath = path.join(process.cwd(), "python_codes", "8_json.py");
      const jsonOutputPath = path.join(jobResultsDir, "audit2_data.json");
      const jsonArgs = [
        jsonScriptPath,
        "--input", jobResultsDir,
        "--out", jsonOutputPath
      ];

      timestamp = new Date().toISOString();
      await storage.addJobLog(job.jobId, `Running: 8_json.py --input ${jobResultsDir} --out ${jsonOutputPath}`);
      console.log(`[JSON Generation] Spawning: 8_json.py`);

      let jsonOut = "";
      let jsonErr = "";

      const jsonExitCode: number = await new Promise((resolve, reject) => {
        const child = spawn(pythonExe, jsonArgs, { cwd: process.cwd() });

        child.stdout.on("data", (chunk) => {
          const s = chunk.toString();
          jsonOut += s;
          storage.addJobLog(job.jobId, s).catch(() => {});
          console.log(`[8_json.py]`, s);
        });

        child.stderr.on("data", (chunk) => {
          const s = chunk.toString();
          jsonErr += s;
          storage.addJobLog(job.jobId, s).catch(() => {});
          console.error(`[8_json.py Error]`, s);
        });

        child.on("error", (e) => reject(e));
        child.on("close", (code) => resolve(code === null ? 1 : code));
      });

      if (jsonExitCode !== 0) {
        executionLogs.push({ 
          script: "8_json.py", 
          status: "error", 
          stdout: jsonOut, 
          stderr: jsonErr, 
          error: `Exit code ${jsonExitCode}`, 
          timestamp 
        });
        await storage.addJobLog(job.jobId, `8_json.py failed with exit code ${jsonExitCode}`);
        // Don't fail the job if JSON generation fails - reports were generated successfully
        await storage.addJobLog(job.jobId, "Reports generated, but JSON generation failed");
      } else {
        executionLogs.push({ 
          script: "8_json.py", 
          status: "success", 
          stdout: jsonOut, 
          stderr: jsonErr, 
          timestamp 
        });
        await storage.addJobLog(job.jobId, "8_json.py completed successfully");
      }

      // ============ STEP 3: Run 6_merge_result.py (merge PDF) AFTER JSON ==========
      try {
        const mergeScriptPath = path.join(process.cwd(), "python_codes", "6_merge_result.py");
        const mergeArgs = [mergeScriptPath, "--input", jobResultsDir];

        timestamp = new Date().toISOString();
        await storage.addJobLog(job.jobId, `Running: 6_merge_result.py --input ${jobResultsDir}`);
        console.log(`[PDF Merge] Spawning: 6_merge_result.py`);

        let mergeOut = "";
        let mergeErr = "";

        const mergeExitCode: number = await new Promise((resolve, reject) => {
          const child = spawn(pythonExe, mergeArgs, { cwd: process.cwd() });

          child.stdout.on("data", (chunk) => {
            const s = chunk.toString();
            mergeOut += s;
            storage.addJobLog(job.jobId, s).catch(() => {});
            console.log(`[6_merge_result.py]`, s);
          });

          child.stderr.on("data", (chunk) => {
            const s = chunk.toString();
            mergeErr += s;
            storage.addJobLog(job.jobId, s).catch(() => {});
            console.error(`[6_merge_result.py Error]`, s);
          });

          child.on("error", (e) => reject(e));
          child.on("close", (code) => resolve(code === null ? 1 : code));
        });

        if (mergeExitCode !== 0) {
          executionLogs.push({
            script: "6_merge_result.py",
            status: "error",
            stdout: mergeOut,
            stderr: mergeErr,
            error: `Exit code ${mergeExitCode}`,
            timestamp
          });
          await storage.addJobLog(job.jobId, `6_merge_result.py failed with exit code ${mergeExitCode}`);
        } else {
          executionLogs.push({
            script: "6_merge_result.py",
            status: "success",
            stdout: mergeOut,
            stderr: mergeErr,
            timestamp
          });
          await storage.addJobLog(job.jobId, "6_merge_result.py completed successfully");
        }
      } catch (mergeErrAny) {
        console.error("PDF merge error:", mergeErrAny);
        await storage.addJobLog(job.jobId, `PDF merge error: ${mergeErrAny}`);
      }

      // ============ STEP 4: Run 7_summary.py (generate summary) AFTER PDF merge ==========
      try {
        const summaryScriptPath = path.join(process.cwd(), "python_codes", "7_summary.py");
        const mergedPdfPath = path.join(jobResultsDir, "Merged_Result.pdf");
        const summaryOutputPath = path.join(jobResultsDir, "Audit_Summary_Report.pdf");
        const summaryArgs = [summaryScriptPath, "--pdf", mergedPdfPath, "--out", summaryOutputPath];

        timestamp = new Date().toISOString();
        await storage.addJobLog(job.jobId, `Running: 7_summary.py --pdf ${mergedPdfPath}`);
        console.log(`[Summary Generation] Spawning: 7_summary.py`);

        let summaryOut = "";
        let summaryErr = "";

        const summaryExitCode: number = await new Promise((resolve, reject) => {
          const child = spawn(pythonExe, summaryArgs, { cwd: process.cwd() });

          child.stdout.on("data", (chunk) => {
            const s = chunk.toString();
            summaryOut += s;
            storage.addJobLog(job.jobId, s).catch(() => {});
            console.log(`[7_summary.py]`, s);
          });

          child.stderr.on("data", (chunk) => {
            const s = chunk.toString();
            summaryErr += s;
            storage.addJobLog(job.jobId, s).catch(() => {});
            console.error(`[7_summary.py Error]`, s);
          });

          child.on("error", (e) => reject(e));
          child.on("close", (code) => resolve(code === null ? 1 : code));
        });

        if (summaryExitCode !== 0) {
          executionLogs.push({
            script: "7_summary.py",
            status: "error",
            stdout: summaryOut,
            stderr: summaryErr,
            error: `Exit code ${summaryExitCode}`,
            timestamp
          });
          await storage.addJobLog(job.jobId, `7_summary.py failed with exit code ${summaryExitCode}`);
        } else {
          executionLogs.push({
            script: "7_summary.py",
            status: "success",
            stdout: summaryOut,
            stderr: summaryErr,
            timestamp
          });
          await storage.addJobLog(job.jobId, "7_summary.py completed successfully");
        }
      } catch (summaryErrAny) {
        console.error("Summary generation error:", summaryErrAny);
        await storage.addJobLog(job.jobId, `Summary generation error: ${summaryErrAny}`);
      }

      // Update job status to done
      await storage.updateJobStatus(job.jobId, "done");
      await storage.addJobLog(job.jobId, "Report, JSON, and summary generation completed successfully");
      console.log(`[Report Generation] Completed for job ${job.jobId}`);
      return executionLogs;

    } catch (bgErr: any) {
      console.error("Report generation runner error:", bgErr);
      try { 
        await storage.updateJobStatus(job.jobId, "failed", bgErr?.message || String(bgErr)); 
      } catch (e) {}
      try { 
        await storage.addJobLog(job.jobId, `Report generation error: ${bgErr?.message || String(bgErr)}`); 
      } catch (e) {}
      return [];
    }
  }

  // Run the summary script (7_summary.py) against the current job's merged PDF
  app.post("/api/run-summary", async (req, res) => {
    try {
      // Allow explicit jobId to be supplied (useful for debugging); otherwise use session
      const suppliedJobId = (req.body && (req.body as any).jobId) || req.query?.jobId;
      const currentJobId = suppliedJobId || req.session.currentJobId;

      if (!currentJobId) {
        return res.status(400).json({ success: false, message: "No active job found" });
      }

      const job = await storage.getJob(currentJobId);
      if (!job || !job.outputPath) {
        return res.status(400).json({ success: false, message: "Job not found or has no output path" });
      }

      const logs: Array<{ type: string; text: string }> = [];

      console.log(`[/api/run-summary] Starting report generation for job: ${currentJobId}`);
      await storage.addJobLog(currentJobId, `Starting report generation`);

      // ===== RUN report.py =====
      const reportScriptPath = path.join(process.cwd(), "python_codes", "report.py");
      const inputDir = job.outputPath;  // Contains rack/, cable/, units/, results/
      const resultsDir = path.join(job.outputPath, "results");  // Save reports to results folder

      // Ensure results directory exists
      if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
      }

      console.log(`[/api/run-summary] Running report.py with input: ${inputDir}, output: ${resultsDir}`);
      await storage.addJobLog(currentJobId, `Running report.py with input: ${inputDir}, output: ${resultsDir}`);

      const reportChild = spawn("python", [reportScriptPath, "--input", inputDir, "--output", resultsDir], { 
        cwd: process.cwd() 
      });

      reportChild.stdout.on("data", (chunk) => {
        const s = chunk.toString();
        logs.push({ type: "stdout", text: s });
        storage.addJobLog(currentJobId, s).catch(() => {});
        console.log("[report.py stdout]", s);
      });

      reportChild.stderr.on("data", (chunk) => {
        const s = chunk.toString();
        logs.push({ type: "stderr", text: s });
        storage.addJobLog(currentJobId, s).catch(() => {});
        console.error("[report.py stderr]", s);
      });

      const reportExitCode: number = await new Promise((resolve, reject) => {
        reportChild.on("error", (e) => reject(e));
        reportChild.on("close", (code) => resolve(code === null ? 1 : code));
      });

      if (reportExitCode !== 0) {
        await storage.addJobLog(currentJobId, `Report script failed with exit code ${reportExitCode}`);
        console.error(`[/api/run-summary] report.py failed with exit code ${reportExitCode}`);
        return res.status(500).json({ success: false, message: "Report generation failed", exitCode: reportExitCode, logs });
      }

      console.log(`[/api/run-summary] report.py completed successfully`);
      await storage.addJobLog(currentJobId, `Report generation completed successfully`);

      // Check if HTML reports were generated
      let reportFiles: string[] = [];
      if (fs.existsSync(resultsDir)) {
        reportFiles = fs.readdirSync(resultsDir).filter(f => f.endsWith('.html'));
        console.log(`[/api/run-summary] Generated report files: ${reportFiles.join(', ')}`);
        await storage.addJobLog(currentJobId, `Generated report files: ${reportFiles.join(', ')}`);
      }

      // ===== RUN 6_merge_result.py =====
      console.log(`[/api/run-summary] Running merge_result.py to create merged PDF`);
      await storage.addJobLog(currentJobId, `Running merge_result.py to create merged PDF`);

      const mergeScriptPath = path.join(process.cwd(), "python_codes", "6_merge_result.py");
      const mergedPdfPath = path.join(resultsDir, "Merged_Result.pdf");

      const mergeChild = spawn("python", [mergeScriptPath, "--input", resultsDir, "--output", mergedPdfPath], {
        cwd: process.cwd()
      });

      mergeChild.stdout.on("data", (chunk) => {
        const s = chunk.toString();
        logs.push({ type: "stdout", text: s });
        storage.addJobLog(currentJobId, s).catch(() => {});
        console.log("[merge.py stdout]", s);
      });

      mergeChild.stderr.on("data", (chunk) => {
        const s = chunk.toString();
        logs.push({ type: "stderr", text: s });
        storage.addJobLog(currentJobId, s).catch(() => {});
        console.error("[merge.py stderr]", s);
      });

      const mergeExitCode: number = await new Promise((resolve, reject) => {
        mergeChild.on("error", (e) => reject(e));
        mergeChild.on("close", (code) => resolve(code === null ? 1 : code));
      });

      if (mergeExitCode !== 0) {
        await storage.addJobLog(currentJobId, `Merge script failed with exit code ${mergeExitCode}`);
        console.error(`[/api/run-summary] merge_result.py failed with exit code ${mergeExitCode}`);
        return res.status(500).json({ success: false, message: "PDF merge failed", exitCode: mergeExitCode, logs });
      }

      console.log(`[/api/run-summary] merge_result.py completed successfully`);
      await storage.addJobLog(currentJobId, `Merged PDF created successfully at ${mergedPdfPath}`);

      // ===== RUN 7_summary.py =====
      console.log(`[/api/run-summary] Running 7_summary.py to generate audit summary`);
      await storage.addJobLog(currentJobId, `Running 7_summary.py to generate audit summary`);

      const summaryScriptPath = path.join(process.cwd(), "python_codes", "7_summary.py");

      const summaryChild = spawn("python", [summaryScriptPath, "--pdf", mergedPdfPath], {
        cwd: process.cwd()
      });

      summaryChild.stdout.on("data", (chunk) => {
        const s = chunk.toString();
        logs.push({ type: "stdout", text: s });
        storage.addJobLog(currentJobId, s).catch(() => {});
        console.log("[summary.py stdout]", s);
      });

      summaryChild.stderr.on("data", (chunk) => {
        const s = chunk.toString();
        logs.push({ type: "stderr", text: s });
        storage.addJobLog(currentJobId, s).catch(() => {});
        console.error("[summary.py stderr]", s);
      });

      const summaryExitCode: number = await new Promise((resolve, reject) => {
        summaryChild.on("error", (e) => reject(e));
        summaryChild.on("close", (code) => resolve(code === null ? 1 : code));
      });

      if (summaryExitCode !== 0) {
        await storage.addJobLog(currentJobId, `Summary script failed with exit code ${summaryExitCode}`);
        console.error(`[/api/run-summary] 7_summary.py failed with exit code ${summaryExitCode}`);
        return res.status(500).json({ success: false, message: "Summary generation failed", exitCode: summaryExitCode, logs });
      }

      console.log(`[/api/run-summary] 7_summary.py completed successfully`);
      await storage.addJobLog(currentJobId, `Summary report generated successfully`);

      // ===== RUN 8_json.py to generate correct JSON structure =====
      console.log(`[/api/run-summary] Running 8_json.py to generate correct JSON structure`);
      await storage.addJobLog(currentJobId, `Running 8_json.py to generate correct JSON structure`);
      
      const jsonScriptPath = path.join(process.cwd(), "python_codes", "8_json.py");
      const jsonOutputPath = path.join(resultsDir, "audit2_data.json");
      const jsonArgs = ["--input", resultsDir, "--out", jsonOutputPath];
      
      const jsonExitCode = await new Promise<number>((resolve) => {
        const jsonProc = spawn("python", [jsonScriptPath, ...jsonArgs]);
        let jsonOutput = "";
        let jsonErrors = "";

        jsonProc.stdout?.on("data", (data) => {
          jsonOutput += data.toString();
          const s = data.toString().trim();
          if (s) console.log(`[8_json.py]`, s);
        });

        jsonProc.stderr?.on("data", (data) => {
          jsonErrors += data.toString();
          const s = data.toString().trim();
          if (s) console.error(`[8_json.py Error]`, s);
        });

        jsonProc.on("close", (code) => {
          if (jsonErrors) {
            console.error(`[8_json.py stderr]: ${jsonErrors}`);
          }
          resolve(code || 0);
        });
      });

      if (jsonExitCode !== 0) {
        console.error(`[/api/run-summary] 8_json.py failed with exit code ${jsonExitCode}`);
        await storage.addJobLog(currentJobId, `8_json.py failed with exit code ${jsonExitCode}`);
        // Don't fail the entire request if JSON generation fails - it's not critical
      } else {
        console.log(`[/api/run-summary] 8_json.py completed successfully`);
        await storage.addJobLog(currentJobId, `8_json.py completed successfully, correct JSON generated`);
      }

      // ===== READ SUMMARY FILES =====
      let summaryText: string | null = null;
      let summaryPdfUrl: string | null = null;
      let auditSummaryPdfUrl: string | null = null;
      let auditJsonUrl: string | null = null;

      // Check for generated summary files in current working directory
      const auditSummaryPdfPath = path.join(process.cwd(), "Audit_Summary_Report.pdf");
      const auditSummaryTxtPath = path.join(process.cwd(), "Audit_Summary_Report.txt");
      const audit2DataJsonPath = path.join(process.cwd(), "audit2_data.json");

      // Copy PDF to results folder
      if (fs.existsSync(auditSummaryPdfPath)) {
        try {
          const destPdf = path.join(resultsDir, "Audit_Summary_Report.pdf");
          fs.copyFileSync(auditSummaryPdfPath, destPdf);
          auditSummaryPdfUrl = `/api/job-file/${currentJobId}/Audit_Summary_Report.pdf`;
          console.log(`[/api/run-summary] Summary PDF copied to results folder`);
        } catch (e) {
          console.error(`[/api/run-summary] Failed to copy summary PDF:`, e);
        }
      }

      // NOTE: Don't copy audit2_data.json from root - 8_json.py already created the correct version in results folder
      // Just set the URL to point to it
      const resultsJsonPath = path.join(resultsDir, "audit2_data.json");
      if (fs.existsSync(resultsJsonPath)) {
        auditJsonUrl = `/api/job-file/${currentJobId}/audit2_data.json`;
        console.log(`[/api/run-summary] Audit JSON available at results folder`);
      }

      // Read and copy summary text
      if (fs.existsSync(auditSummaryTxtPath)) {
        try {
          summaryText = fs.readFileSync(auditSummaryTxtPath, { encoding: "utf-8" });
          // Also copy to results folder so /api/report/:id/summary can find it
          const destTxt = path.join(resultsDir, "Audit_Summary_Report.txt");
          fs.copyFileSync(auditSummaryTxtPath, destTxt);
          console.log(`[/api/run-summary] Summary text read successfully and copied to results folder`);
        } catch (e) {
          console.error(`[/api/run-summary] Failed to read/copy summary text:`, e);
        }
      }

      res.json({ 
        success: true, 
        message: "Report generation, PDF merge, and summary completed successfully",
        jobId: currentJobId,
        reportFiles,
        summaryText,
        summaryPdfUrl: auditSummaryPdfUrl,
        auditJsonUrl,
        mergedPdfUrl: `/api/job-file/${currentJobId}/Merged_Result.pdf`,
        logs 
      });
    } catch (error: any) {
      console.error("Run-summary error:", error);
      res.status(500).json({ success: false, message: error.message || "Run summary failed" });
    }
  });

  // Serve job output files (PDFs, reports, etc)
  app.get("/api/job-file/:jobId/:filename", async (req, res) => {
    try {
      const { jobId, filename } = req.params;

      // Get job from database or filesystem
      let job = await storage.getJob(jobId);
      
      if (!job) {
        // Try to find from filesystem
        const jobsOutputDir = path.join(process.cwd(), "jobs_output");
        const jobDir = path.join(jobsOutputDir, jobId, "output");
        
        if (fs.existsSync(jobDir)) {
          job = {
            jobId: jobId,
            outputPath: jobDir
          };
        } else {
          return res.status(404).json({ success: false, message: "Job not found" });
        }
      }

      // Security: only allow files from the results directory
      const allowedDirs = [
        path.join(job.outputPath, "results"),
        path.join(job.outputPath, "reports")
      ];

      const filePath = path.join(job.outputPath, "results", filename);
      
      // Verify the file is within allowed directories
      if (!allowedDirs.some(dir => filePath.startsWith(dir))) {
        return res.status(403).json({ success: false, message: "Access denied" });
      }

      if (!fs.existsSync(filePath)) {
        return res.status(404).json({ success: false, message: "File not found" });
      }

      // Serve the file
      res.download(filePath);
    } catch (error: any) {
      console.error("job-file error:", error);
      res.status(500).json({ success: false, message: error.message || "Failed to retrieve file" });
    }
  });

  app.get("/api/current-job", async (req, res) => {
    try {
      let currentJobId = req.session.currentJobId;
      let job = null;

      console.log(`[/api/current-job] Current session jobId: ${currentJobId}`);

      if (currentJobId) {
        job = await storage.getJob(currentJobId);
      }

      // If no current job or job not found, get the latest job for the user
      if (!job && req.session.userId) {
        const userJobs = await storage.getUserJobs(req.session.userId);
        console.log(`[/api/current-job] Found ${userJobs.length} database jobs for user`);
        if (userJobs.length > 0) {
          // Sort by createdAt descending to get the latest
          userJobs.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
          job = userJobs[0];
          // Set it as current job
          req.session.currentJobId = job.jobId;
          console.log(`[/api/current-job] Using latest database job: ${job.jobId}`);
        }
      }

      // If still no job found in database, check filesystem for latest job directory
      if (!job) {
        console.log(`[/api/current-job] No database job found, checking filesystem...`);
        try {
          const jobsOutputDir = path.join(process.cwd(), "jobs_output");
          if (fs.existsSync(jobsOutputDir)) {
            const jobDirs = fs.readdirSync(jobsOutputDir, { withFileTypes: true })
              .filter(d => d.isDirectory() && d.name.startsWith("job_"))
              .map(d => ({
                name: d.name,
                time: fs.statSync(path.join(jobsOutputDir, d.name)).mtime.getTime()
              }))
              .sort((a, b) => b.time - a.time);
            
            if (jobDirs.length > 0) {
              const latestJobName = jobDirs[0].name;
              const outputPath = path.join(jobsOutputDir, latestJobName, "output");
              
              // Check if output directory exists and has segmentation results
              if (fs.existsSync(outputPath)) {
                job = {
                  jobId: latestJobName,
                  userId: req.session.userId || "unknown",
                  outputPath: outputPath,
                  status: "completed",
                  createdAt: new Date(jobDirs[0].time).toISOString(),
                  fromFilesystem: true  // Flag indicating this is from filesystem, not database
                };
                req.session.currentJobId = latestJobName;
                console.log(`[/api/current-job] Found filesystem job: ${latestJobName} at ${outputPath}`);
              }
            }
          }
        } catch (fsError) {
          console.error("[/api/current-job] Error checking filesystem:", fsError);
        }
      }

      console.log(`[/api/current-job] Returning job: ${job?.jobId || "null"}, outputPath: ${job?.outputPath || "none"}`);
      res.json({ success: true, job });
    } catch (error) {
      console.error("Error fetching current job:", error);
      res.status(500).json({ success: false, message: "Failed to fetch current job" });
    }
  });

  app.post("/api/generate-report", async (req, res) => {
    try {
      console.log("[Report Generation] Starting report generation process");

      const currentJobId = req.session.currentJobId;
      if (!currentJobId) {
        return res.status(400).json({ success: false, message: "No active job found" });
      }

      const job = await storage.getJob(currentJobId);
      if (!job || !job.outputPath) {
        return res.status(400).json({ success: false, message: "Job not found or has no output path" });
      }

      const jobResultsDir = path.join(job.outputPath, "Results");
      // Python script outputs class folders directly to outputPath
      const segmentedOutputDir = job.outputPath;

      if (!fs.existsSync(jobResultsDir)) {
        fs.mkdirSync(jobResultsDir, { recursive: true });
      }

      const pdfPath = path.join(jobResultsDir, "Merged_Result.pdf");

      // If a PDF already exists for this job and client didn't request a forced regen,
      // return immediately so the frontend can display it without waiting for regeneration.
      const forceRegen = req.query?.force === "true" || req.body?.force === true;
      if (fs.existsSync(pdfPath) && !forceRegen) {
        console.log("[Report Generation] PDF already exists; returning existing PDF info");
        const pdfUrl = `/api/report/${job.jobId}/pdf`;
        return res.json({
          success: true,
          message: "Report already generated",
          jobId: job.jobId,
          pdfPath: "Results/Merged_Result.pdf",
          pdfUrl,
        });
      }

      // If forcing regeneration, remove existing PDF so it will be recreated.
      if (fs.existsSync(pdfPath) && forceRegen) {
        try {
          fs.unlinkSync(pdfPath);
          console.log("[Report Generation] Cleared existing PDF (force)");
        } catch (e) {
          console.warn("[Report Generation] Failed to remove existing PDF:", e);
        }
      }

      const scriptConfigs = [
        { script: "1_rack_match.py", inputSubfolder: "rack" },
        { script: "2_switch_match.py", inputSubfolder: "switch" },
        { script: "3_patchpanel_match.py", inputSubfolder: "patch_panel" },
        { script: "4_1_conneted_port_match.py", inputSubfolder: "connected_port" },
        { script: "4_port_match.py", inputSubfolder: null },
        { script: "5_cable_match.py", inputSubfolder: "cables" },
        { script: "6_merge_result.py", inputSubfolder: null },
        { script: "7_summary.py", inputSubfolder: null },
        { script: "8_json.py", inputSubfolder: null },
      ];

      const executionLogs: Array<{
        script: string;
        status: "success" | "error";
        stdout: string;
        stderr: string;
        error?: string;
        timestamp: string;
      }> = [];

      // Run scripts in background so the HTTP request doesn't block and trigger 504s.
      // Use the helper below to run the report generation asynchronously.
      runReportForJob(job).catch((err) => {
        console.error("Background runReportForJob error:", err);
      });

      // Immediately return job info so client can fetch the job-specific PDF endpoint
      const pdfUrl = `/api/report/${job.jobId}/pdf`;

      res.json({
        success: true,
        message: "Report generation started",
        jobId: job.jobId,
        pdfPath: "Results/Merged_Result.pdf",
        pdfUrl,
        logs: executionLogs,
      });
    } catch (error: any) {
      console.error("Report generation error:", error);
      res.status(500).json({
        success: false,
        message: error.message || "Report generation failed",
      });
    }
  });

  app.get("/api/report/pdf", async (req, res) => {
    try {
      const currentJobId = req.session.currentJobId;
      if (!currentJobId) {
        return res.status(400).json({ success: false, message: "No active job found" });
      }

      const job = await storage.getJob(currentJobId);
      if (!job || !job.outputPath) {
        return res.status(400).json({ success: false, message: "Job not found" });
      }

      const pdfPath = path.join(job.outputPath, "Results", "Merged_Result.pdf");

      if (!fs.existsSync(pdfPath)) {
        return res.status(404).json({ success: false, message: "Report PDF not found" });
      }

      res.setHeader("Content-Type", "application/pdf");
      res.setHeader("Content-Disposition", 'inline; filename="Merged_Result.pdf"');
      res.sendFile(pdfPath);
    } catch (error) {
      console.error("Error serving PDF:", error);
      res.status(500).json({ success: false, message: "Failed to serve PDF" });
    }
  });

  app.get("/api/user/reports", async (req, res) => {
    try {
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const reports = await storage.getUserReports(userId);
      res.json({ success: true, reports });
    } catch (error) {
      console.error("Get reports error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch reports" });
    }
  });

  app.get("/api/user/reports/:id/pdf", async (req, res) => {
    try {
      const reportId = req.params.id;
      const report = await storage.getReport(reportId);

      if (!report) {
        return res.status(404).json({ success: false, message: "Report not found" });
      }

      const pdfPath = path.join(process.cwd(), report.pdfPath);

      if (!fs.existsSync(pdfPath)) {
        return res.status(404).json({ success: false, message: "PDF file not found" });
      }

      res.setHeader("Content-Type", "application/pdf");
      res.setHeader("Content-Disposition", `inline; filename="${report.filename}"`);
      res.sendFile(pdfPath);
    } catch (error) {
      console.error("Error serving report PDF:", error);
      res.status(500).json({ success: false, message: "Failed to serve PDF" });
    }
  });

  app.delete("/api/user/reports/:id", async (req, res) => {
    try {
      const reportId = req.params.id;
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const report = await storage.getReport(reportId);
      if (!report) {
        return res.status(404).json({ success: false, message: "Report not found" });
      }

      if (report.userId !== userId) {
        return res.status(403).json({ success: false, message: "Not authorized to delete this report" });
      }

      const pdfPath = path.join(process.cwd(), report.pdfPath);
      if (fs.existsSync(pdfPath)) {
        fs.unlinkSync(pdfPath);
      }

      await storage.deleteReport(reportId);

      res.json({ success: true, message: "Report deleted successfully" });
    } catch (error) {
      console.error("Delete report error:", error);
      res.status(500).json({ success: false, message: "Failed to delete report" });
    }
  });

  app.get("/api/history/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const sessionUserId = req.session?.userId;

      if (!sessionUserId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      if (sessionUserId !== userId) {
        return res.status(403).json({ success: false, message: "Not authorized" });
      }

      const userJobs = await storage.getUserJobs(userId);
      
      const reports: Array<{
        id: string;
        userId: string;
        title: string;
        filename: string;
        pdfPath: string;
        processedImage?: string;
        createdAt: string;
      }> = [];

      for (const job of userJobs) {
        if (job.status === "done" && job.outputPath) {
          const pdfPath = path.join(job.outputPath, "Results", "Merged_Result.pdf");
          
          if (fs.existsSync(pdfPath)) {
            let processedImage: string | undefined;
            const outputDir = job.outputPath;
            // Support both naming conventions
            const classFolders = ["rack", "switch", "patch_panel", "cables", "cable", "connected_port", "empty_port", "units"];
            
            for (const folder of classFolders) {
              const folderPath = path.join(outputDir, folder);
              if (fs.existsSync(folderPath)) {
                const files = fs.readdirSync(folderPath);
                const imageFile = files.find(f => /\.(jpg|jpeg|png|webp)$/i.test(f));
                if (imageFile) {
                  processedImage = path.join("jobs_output", job.jobId, "output", folder, imageFile).replace(/\\/g, "/");
                  break;
                }
              }
            }

            reports.push({
              id: job.jobId,
              userId: job.userId || userId,
              title: "Merged Result",
              filename: "Merged_Result.pdf",
              pdfPath: pdfPath,
              processedImage,
              createdAt: job.createdAt ? new Date(job.createdAt).toISOString() : new Date().toISOString(),
            });
          }
        }
      }

      res.json({ success: true, reports });
    } catch (error) {
      console.error("Get history error:", error);
      res.status(500).json({ success: false, message: "Failed to fetch history" });
    }
  });

  app.get("/api/report/:id/pdf", async (req, res) => {
    try {
      const jobId = req.params.id;
      const sessionUserId = req.session?.userId;

      if (!sessionUserId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const job = await storage.getJob(jobId);

      if (!job || !job.outputPath) {
        return res.status(404).json({ success: false, message: "Report not found" });
      }

      if (job.userId !== sessionUserId) {
        console.warn(`[PDF ACCESS DENIED] jobId=${jobId} job.userId=${job.userId} sessionUserId=${sessionUserId}`);
        return res.status(403).json({ success: false, message: "Not authorized to view this report" });
      }

      const pdfPath = path.join(job.outputPath, "Results", "Merged_Result.pdf");

      if (!fs.existsSync(pdfPath)) {
        return res.status(404).json({ success: false, message: "PDF file not found" });
      }

      res.setHeader("Content-Type", "application/pdf");
      res.setHeader("Content-Disposition", 'inline; filename="Merged_Result.pdf"');
      res.sendFile(pdfPath);
    } catch (error) {
      console.error("Error serving report PDF:", error);
      res.status(500).json({ success: false, message: "Failed to serve PDF" });
    }
  });

  app.get("/api/report/:id/summary", async (req, res) => {
    try {
      const jobId = req.params.id;
      const job = await storage.getJob(jobId);
      if (!job || !job.outputPath) return res.status(404).json({ success: false, message: "Job not found" });

      // Try lowercase "results" first, then "Results"
      let txtPath = path.join(job.outputPath, "results", "Audit_Summary_Report.txt");
      if (!fs.existsSync(txtPath)) {
        txtPath = path.join(job.outputPath, "Results", "Audit_Summary_Report.txt");
      }
      
      if (!fs.existsSync(txtPath)) {
        return res.status(404).json({ success: false, message: "Summary text not found" });
      }

      const content = fs.readFileSync(txtPath, { encoding: "utf-8" });
      res.json({ success: true, summary: content });
    } catch (err) {
      console.error("Error serving summary text:", err);
      res.status(500).json({ success: false, message: "Failed to serve summary" });
    }
  });

  app.get("/api/report/:id/json", async (req, res) => {
    try {
      const jobId = req.params.id;
      const job = await storage.getJob(jobId);

      if (!job || !job.outputPath) {
        return res.status(404).json({ success: false, message: "Job not found" });
      }

      const jsonPath = path.join(job.outputPath, "Results", "audit2_data.json");

      if (!fs.existsSync(jsonPath)) {
        return res.status(404).json({
          success: false,
          message: "JSON report not found"
        });
      }

      const jsonContent = fs.readFileSync(jsonPath, 'utf-8');
      const jsonData = JSON.parse(jsonContent);

      res.json(jsonData);
    } catch (err) {
      console.error("Error serving JSON report:", err);
      res.status(500).json({
        success: false,
        message: "Failed to serve JSON report"
      });
    }
  });

  // Debug endpoint: return job info and PDF existence for troubleshooting
  app.get("/api/debug/job/:id", async (req, res) => {
    try {
      const jobId = req.params.id;
      const sessionUserId = req.session?.userId || null;
      const job = await storage.getJob(jobId);

      if (!job) {
        return res.status(404).json({ success: false, message: "Job not found" });
      }

      const pdfPath = job.outputPath ? path.join(job.outputPath, "Results", "Merged_Result.pdf") : null;
      const pdfExists = pdfPath ? fs.existsSync(pdfPath) : false;

      res.json({
        success: true,
        job: {
          jobId: job.jobId,
          userId: job.userId,
          status: job.status,
          inputPath: job.inputPath,
          outputPath: job.outputPath,
          createdAt: job.createdAt,
        },
        sessionUserId,
        pdfPath,
        pdfExists,
      });
    } catch (err) {
      console.error("Debug job endpoint error:", err);
      res.status(500).json({ success: false, message: "Debug endpoint error" });
    }
  });

  app.delete("/api/reports/:id", async (req, res) => {
    try {
      const jobId = req.params.id;
      const userId = req.session?.userId;

      if (!userId) {
        return res.status(401).json({ success: false, message: "Not authenticated" });
      }

      const job = await storage.getJob(jobId);
      if (!job) {
        return res.status(404).json({ success: false, message: "Report not found" });
      }

      if (job.userId !== userId) {
        return res.status(403).json({ success: false, message: "Not authorized to delete this report" });
      }

      if (job.outputPath) {
        const pdfPath = path.join(job.outputPath, "Results", "Merged_Result.pdf");
        if (fs.existsSync(pdfPath)) {
          fs.unlinkSync(pdfPath);
        }
      }

      res.json({ success: true, message: "Report deleted successfully" });
    } catch (error) {
      console.error("Delete report error:", error);
      res.status(500).json({ success: false, message: "Failed to delete report" });
    }
  });

  const server = createServer(app);
  return server;
}
