import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
 
import re
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
from rapidfuzz import fuzz
 
import argparse
from vendor_ocr import detect_vendor

# -------------------- CLI / ARGPARSE --------------------
parser = argparse.ArgumentParser(description='Switch match report generator')
parser.add_argument('--input', '-i', help='Directory with test images')
parser.add_argument('--ref-emb', '-r', help='Directory containing reference .npy embeddings')
parser.add_argument('--output', '-o', help='Output HTML file path or directory')
parser.add_argument('--model', '-m', help='YOLO model path')
parser.add_argument('--desc', '-d', help='Description Excel file')
parser.add_argument('--device', choices=['cpu','cuda'], help='Force device selection')
parser.add_argument('--job', '-J', help='Job folder path (will use job/output/Results for output if provided)')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
args = parser.parse_args()

candidate_results = None

# (CLI overrides applied later after defaults to ensure sensible defaults remain)
 
# ================= CONFIG =================
TEST_IMAGE_DIR = r"jobs_output\job_1767354311550_e2907381\output\switch"
REFERENCE_EMB_DIR = r"Switch_emb"
OUTPUT_HTML = r"switch_match_report.html"
 
MODEL_PATH = r"Trained_Models\port_count.pt"
 
ROW_TOLERANCE = 15
MODEL_NAME = "openai/clip-vit-base-patch32"
 
DESCRIPTION_XLSX = r"Description_excel\Merged_descritpion_switch.xlsx"
MODEL_COL = "Model_Name"
DESC_COL = "Description"
FUZZY_THRESHOLD = 85
# =========================================
 
# Apply CLI overrides (if any) after defaults to ensure overrides take effect
if args.input:
    TEST_IMAGE_DIR = args.input
if args.ref_emb:
    REFERENCE_EMB_DIR = args.ref_emb
if args.output:
    OUTPUT_HTML = args.output
if args.model:
    MODEL_PATH = args.model
if args.desc:
    DESCRIPTION_XLSX = args.desc

# If job folder specified, prefer job/output/switch for images and job/output/Results for output
if args.job:
    job_root = os.path.abspath(args.job)
    cand_in = os.path.join(job_root, 'output', 'switch')
    cand_results = os.path.join(job_root, 'output', 'Results')
    if os.path.exists(cand_in):
        TEST_IMAGE_DIR = cand_in
    candidate_results = cand_results

device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
try:
    yolo_model = YOLO(MODEL_PATH)
except Exception:
    # keep original behavior: attempt load regardless
    yolo_model = YOLO(MODEL_PATH)
 
# ================= UTILS =================
def normalize_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())
 
 
def normalize_vendor(s):
    return normalize_text(s)
 
 
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
 
 
def nearest_ports(detected):
    candidates = [8, 12, 16, 20, 24, 28, 48]
    distances = [(abs(p - detected), p) for p in candidates]
    distances.sort()
    return [p for _, p in distances[:3]]
 
 
# ================= DESCRIPTION LOADING (ROBUST) =================
desc_df = pd.read_excel(DESCRIPTION_XLSX)
 
desc_map = []
for _, row in desc_df.iterrows():
    models = str(row[MODEL_COL]).split("/")
    for m in models:
        norm = normalize_text(m)
        if norm:
            desc_map.append({
                "norm_model": norm,
                "description": row[DESC_COL]
            })
 
 
def get_description(model_name):
    norm = normalize_text(model_name)
    best_score = 0
    best_desc = ""
 
    for item in desc_map:
        score = fuzz.ratio(norm, item["norm_model"])
        if score > best_score:
            best_score = score
            best_desc = item["description"]
 
    return best_desc if best_score >= FUZZY_THRESHOLD else ""
 
 
# ================= PORT EXTRACTION =================
def extract_port_count(name):
    name = os.path.splitext(name)[0].upper()
 
    m = re.search(r"-(\d{2})(?:[A-Z]|$)", name)
    if m:
        return int(m.group(1))
 
    m = re.search(r"(?:GS|DGS|SGD|XS|XGS)\D*(\d{2})$", name)
    if m:
        return int(m.group(1))
 
    m = re.search(r"\d{2}(\d{2})$", name)
    if m:
        return int(m.group(1))
 
    return None
 
 
# ================= SEARCH TIERS (VENDOR KNOWN) =================
def port_search_tiers(detected):
    if detected <= 8:
        return [[8], [12], [16], [20], [24], [28], [48]]
    if detected <= 12:
        return [[12], [16], [20], [24], [28], [48]]
    if detected <= 16:
        return [[16, 20], [24], [28], [48]]
    if detected <= 20:
        return [[20], [24], [28], [48]]
    return [[24], [28], [48]]
 
 
# ================= CLIP =================
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()
 
 
# ================= YOLO ANALYSIS =================
def analyze_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
 
    h, w, _ = img.shape
    results = yolo_model(img, conf=0.35)
 
    ports = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        ports.append(((x1 + x2) / 2, (y1 + y2) / 2))
 
    if not ports:
        return None
 
    ports = sorted(ports, key=lambda p: p[1])
 
    rows = []
    for p in ports:
        placed = False
        for r in rows:
            row_center = np.median([pt[1] for pt in r])
            if abs(row_center - p[1]) < ROW_TOLERANCE:
                r.append(p)
                placed = True
                break
        if not placed:
            rows.append([p])
 
    left = sum(1 for x, _ in ports if x < 0.5 * w)
    right = len(ports) - left
 
    arrangement = "CENTER"
    if left / len(ports) >= 0.6:
        arrangement = "LEFT"
    elif right / len(ports) >= 0.6:
        arrangement = "RIGHT"
 
    return {
        "num_ports": len(ports),
        "num_rows": len(rows),
        "arrangement": arrangement
    }
 
 
# ================= LOAD REFERENCES =================
def load_reference_models(vendor=None):
    models = []
 
    vendor_dirs = [
        d for d in os.listdir(REFERENCE_EMB_DIR)
        if os.path.isdir(os.path.join(REFERENCE_EMB_DIR, d))
    ]
 
    if vendor:
        vmap = {normalize_vendor(v): v for v in vendor_dirs}
        key = normalize_vendor(vendor)
        vendor_dirs = [vmap[key]] if key in vmap else []
 
    for v in vendor_dirs:
        path = os.path.join(REFERENCE_EMB_DIR, v)
        for f in os.listdir(path):
            if f.endswith(".npy"):
                data = np.load(os.path.join(path, f), allow_pickle=True).item()
                models.append({
                    "filename": f,
                    "vendor": v,
                    "embedding": data["embedding"],
                    "rows": data.get("num_rows"),
                    "arrangement": data.get("arrangement")
                })
 
    return models
 
 
# ================= MATCH LOGIC =================
def match_model(test_emb, info):
    detected_ports = info.get("num_ports")
    detected_rows = info.get("num_rows")
    arrangement = info.get("arrangement")
    vendor = info.get("vendor")
 
    # ------------------------------
    # 1️⃣ Load references
    # ------------------------------
    refs = load_reference_models(vendor) if vendor else load_reference_models(None)
 
    if not refs:
        return None
 
    # ------------------------------
    # 2️⃣ Try structured matching (best effort)
    # ------------------------------
    candidates = refs
 
    # --- Port filtering (ONLY if detected) ---
    if detected_ports:
        ports = nearest_ports(detected_ports)
        port_filtered = [
            r for r in candidates
            if extract_port_count(r["filename"]) in ports
        ]
        if port_filtered:
            candidates = port_filtered
 
    # --- Row filtering ---
    if detected_rows:
        row_filtered = [
            r for r in candidates
            if r.get("rows") == detected_rows
        ]
        if row_filtered:
            candidates = row_filtered
 
    # --- Arrangement filtering ---
    if arrangement:
        arr_filtered = [
            r for r in candidates
            if r.get("arrangement") == arrangement
        ]
        if arr_filtered:
            candidates = arr_filtered
 
    # ------------------------------
    # 3️⃣ FINAL FALLBACK (CRITICAL)
    # ------------------------------
    # Even if all filters fail, we STILL match by CLIP
    best = max(
        candidates,
        key=lambda r: cosine(test_emb, r["embedding"])
    )
 
    return best
 
 
 
# ================= HTML =================
def generate_html(rows):
    html = """
<!DOCTYPE html>
<html>
<head>
<style>
body { font-family: Arial; background: #f5f5f5; }
table { border-collapse: collapse; width: 100%; background: white; }
th, td { padding: 12px; border: 1px solid #ddd; text-align: center; }
th { background: #2c3e50; color: white; }
img { max-width: 220px; border-radius: 6px; }
</style>
</head>
<body>
<h2>Switch Analysis Report</h2>
<table>
<tr>
<th>Switch</th>
<th>Predicted Vendor</th>
<th>Switch Model</th>
<th>Description</th>
<th>Detected OCR</th>
</tr>
"""
 
    for r in rows:
        html += f"""
<tr>
<td><img src="{r['test']}"></td>
<td>{r['vendor']}</td>
<td>{r['model']}</td>
<td>{r['description']}</td>
<td>{r['ocr_text']}</td>
</tr>
"""
 
    html += "</table></body></html>"
 
    out_path = OUTPUT_HTML or "Results/2_switch_classification.html"
    # If a job results dir is provided and out_path is a bare filename, place it there
    if candidate_results and (not os.path.isabs(out_path)) and not os.path.dirname(out_path):
        out_path = os.path.join(candidate_results, out_path)

    # Ensure out_path is a file path (not a directory). If a directory path
    # is provided, place the default filename inside it. Also ensure .html ext.
    default_name = '2_switch_classification.html'
    try:
        if os.path.isdir(out_path) or out_path.endswith(os.sep):
            out_path = os.path.join(out_path, default_name)
        base = os.path.basename(out_path)
        if os.path.splitext(base)[1] == '':
            out_path = out_path + '.html'
    except Exception:
        out_path = os.path.join('Results', default_name)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
 
 
# ================= RUN =================
rows = []

if not TEST_IMAGE_DIR or not os.path.exists(TEST_IMAGE_DIR):
    print(f"[2_switch_match] ERROR: input folder not found: {TEST_IMAGE_DIR}")
    print("No switch images to process. Exiting.")
    exit(1)

for f in os.listdir(TEST_IMAGE_DIR):
    if not f.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
 
    path = os.path.join(TEST_IMAGE_DIR, f)
 
    info = analyze_image(path)
    if not info:
        continue
 
    vendor_result = detect_vendor(path)
    if not vendor_result:
        continue
    info["vendor"] = vendor_result.get("matched_vendor")
    emb = get_embedding(path)
 
    best = match_model(emb, info)
    if not best:
        continue
 
    rows.append({
        "test": "file:///" + path.replace("\\", "/"),
        "vendor": best["vendor"],
        "model": os.path.splitext(best["filename"])[0],
        "description": get_description(os.path.splitext(best["filename"])[0]),
        "ocr_text": vendor_result.get("ocr_used_text", "")
    })
 
out_html = generate_html(rows)
print("✅ HTML REPORT GENERATED:", out_html)
 
 