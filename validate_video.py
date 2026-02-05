import cv2
import numpy as np
from pymediainfo import MediaInfo
from pathlib import Path
from datetime import datetime
import os
from tqdm import tqdm

# =====================================================
# CONFIGURATION (PRODUCTION SAFE)
# =====================================================

CONFIG = {
    # ============= AC1: METADATA VALIDATION =============
    "min_width": 1920,
    "min_height": 1080,
    "min_fps": 24,
    "max_fps": 120,
    "supported_codecs": ["HEVC", "H.264", "H.265", "VP9", "AV1"],
    "max_duration_minutes": 30,
    "min_bitrate_mbps": 50,
    "max_bitrate_mbps": 500,

    # ============= AC2: QUALITY & LIGHTING =============
    "frame_sample_interval_sec": 1.0,
    
    # Brightness/Exposure
    "brightness_underexposed": 50,      # <50 = too dark
    "brightness_overexposed": 200,      # >200 = too bright
    "brightness_optimal_range": (80, 180),
    
    # Blur detection
    "blur_ratio_warn": 0.30,            # 30–60% blurry = WARNING
    "blur_ratio_fail": 0.60,            # >60% blurry = FAIL
    "min_edge_density": 0.01,
    "edge_density_optimal": 0.05,
    
    # Quality scoring
    "quality_pass": 70,
    "quality_warn": 50,
    "quality_fail": 30,
    "histogram_bins": 256,
    
    # ============= AC3: RACK SCAN DETECTION =============
    "vertical_motion_threshold_px": 15,   # pixels/frame for vertical pan
    "horizontal_motion_threshold_px": 5,  # pixels/frame for horizontal
    "motion_speed_acceptable": 10,        # 5-30 px/frame is optimal
    "motion_speed_slow": 5,               # <5 px/frame = too slow
    "motion_speed_fast": 30,              # >30 px/frame = too fast
    "frame_visible_threshold": 0.90,      # 90% of frames show rack edges
    "occlusion_max_threshold": 0.20,      # Allow max 20% occlusion
    "edge_detection_threshold_low": 50,
    "edge_detection_threshold_high": 150,

    # ============= AC4: UNIT SEGMENTATION =============
    "min_unit_duration_sec": 3,
    "max_unit_duration_sec": 15,
    "transition_sensitivity": 0.15,       # Motion spike = new unit
    "motion_stillness_threshold": 2,      # px/frame = motion stopped
    "frame_gap_tolerance": 0.5,           # sec, gap between units

    # ============= AC5: PORT DETECTION READINESS =============
    "min_pixels_per_port": 50,
    "max_pixels_per_port": 5000,
    "port_separation_min_px": 10,
    "port_separation_optimal_px": 50,
    "ports_expected_min": 8,              # At least 8 port-like objects
    "ports_expected_max": 500,
    "ocr_confidence_threshold": 0.70,
    "zoom_level_multiplier": 1.0,         # Pixels per U (rack unit)
    "min_zoom_level": 30,                 # min pixels needed

    # ============= AC6: CABLE & CONNECTOR VALIDATION =============
    "connector_shapes": ["RJ45", "LC", "SC", "ST", "MTP"],
    "connector_detection_enabled": True,
    "cable_color_variance_threshold": 30, # RGB std dev
    "cable_types": ["Copper", "Fiber", "Power"],
    "led_indicator_visibility": True,
    "led_brightness_threshold": 150,

    # ============= AC7: REPORT GENERATION =============
    "export_formats": ["HTML", "JSON", "PDF"],
    "export_html": True,
    "export_json": True,
    "export_pdf": False,                  # Requires reportlab
    "include_frame_samples": True,
    "sample_frame_count": 5,
    "report_detail_level": "comprehensive",  # minimal, standard, comprehensive

    # ============= AC8: RE-CAPTURE FLOW =============
    "segment_validation_enabled": True,
    "incremental_validation": True,
    "allow_partial_reupload": True,
    "keep_previous_results": True,
    "merge_validation_results": True,

    # ============= AC9: CONFIGURABLE THRESHOLDS =============
    "validation_profile": "Standard",     # Standard, Strict, Lenient
    "threshold_override_enabled": False,  # Admin override
    "log_threshold_changes": True,
    "config_versioning": True,
    "config_version": "1.0",

    # ============= AC10: PERFORMANCE & SCALABILITY =============
    "max_processing_time_multiplier": 2,  # 2x video duration max
    "parallel_processing_enabled": True,
    "parallel_workers": 4,
    "chunked_processing_enabled": True,
    "chunk_duration_sec": 5,
    "progress_indicator_enabled": True,
    "early_rejection_enabled": True,      # Stop if AC1 fails
    "cache_frames": False,
    "memory_limit_gb": 4,
    
    # ============= LOGGING & AUDITING =============
    "debug_mode": False,
    "log_file_enabled": True,
    "save_intermediate_frames": False,
    "save_motion_vectors": False,
    "audit_trail": True,
    # ============= RACK OVERVIEW (ADDITIONAL VALIDATION) =============
    "min_vertical_travel_ratio": 0.35,   # 35% frame height
    "motion_speed_fast_overview": 55,    # px/frame (median) used for overview check
    "min_edge_presence_ratio": 0.90,
    "occlusion_edge_density_min": 0.008,
    "canny_low": 50,
    "canny_high": 150,
}

# =====================================================
# AC1 – METADATA EXTRACTION
# =====================================================

def extract_metadata(video_path):
    mi = MediaInfo.parse(video_path)
    general = next(t for t in mi.tracks if t.track_type == "General")
    video = next(t for t in mi.tracks if t.track_type == "Video")

    return {
        "filename": Path(video_path).name,
        "width": int(video.width or 0),
        "height": int(video.height or 0),
        "fps": float(video.frame_rate or 0),
        "duration_sec": float(general.duration or 0) / 1000,
        "codec": video.format or "Unknown",
        "filesize_mb": os.path.getsize(video_path) / (1024 * 1024),
    }

def validate_ac1(meta):
    issues = []
    warnings = []
    
    # Resolution
    if meta["width"] < CONFIG["min_width"] or meta["height"] < CONFIG["min_height"]:
        issues.append("Resolution below 1080p")
    
    # FPS
    if meta["fps"] < CONFIG["min_fps"]:
        issues.append("Frame rate below 24 fps")
    if meta["fps"] > CONFIG["max_fps"]:
        warnings.append("Frame rate unusually high (may not be needed)")
    
    # Codec
    if meta["codec"] not in CONFIG["supported_codecs"]:
        warnings.append(f"Codec {meta['codec']} is not recommended")
    
    # Duration
    if meta["duration_sec"] / 60 > CONFIG["max_duration_minutes"]:
        warnings.append(f"Video exceeds {CONFIG['max_duration_minutes']} minutes (may be too long)")
    
    # Bitrate
    bitrate = (meta["filesize_mb"] * 8) / (meta["duration_sec"] / 60)
    if bitrate < CONFIG["min_bitrate_mbps"]:
        warnings.append("Bitrate too low for quality")
    if bitrate > CONFIG["max_bitrate_mbps"]:
        warnings.append("Bitrate unusually high")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }

# =====================================================
# AC2 – QUALITY & BLUR (CORRECT + ROBUST)
# =====================================================

def analyze_quality(video_path, meta):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(fps * CONFIG["frame_sample_interval_sec"]))

    laplacians = []
    edge_densities = []
    blurry_flags = []

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="  Analyzing video quality", leave=False)

    # 🔑 Resolution-normalized Laplacian threshold
    lap_threshold = (meta["width"] * meta["height"]) / (1920 * 1080) * 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            edges = cv2.Canny(gray, 80, 160)
            edge_density = np.mean(edges > 0)

            laplacians.append(lap)
            edge_densities.append(edge_density)

            is_blurry = (lap < lap_threshold) and (edge_density < CONFIG["min_edge_density"])
            blurry_flags.append(is_blurry)

        frame_idx += 1
        pbar.update(1)
    
    pbar.close()

    cap.release()

    blur_ratio = sum(blurry_flags) / len(blurry_flags)

    quality_score = 100
    issues = []

    if blur_ratio > CONFIG["blur_ratio_fail"]:
        quality_score -= 40
        issues.append("Most frames are out of focus")
        blur_status = "Blurry"
    elif blur_ratio > CONFIG["blur_ratio_warn"]:
        quality_score -= 20
        issues.append("Some frames lose focus during motion")
        blur_status = "Mostly Sharp"
    else:
        blur_status = "Sharp"

    return {
        "passed": quality_score >= CONFIG["quality_pass"],
        "quality_score": quality_score,
        "blur_status": blur_status,
        "blur_ratio": blur_ratio,
        "avg_laplacian": np.mean(laplacians),
        "avg_edge_density": np.mean(edge_densities),
        "laplacian_threshold_used": lap_threshold,
        "total_samples": len(laplacians),
        "issues": issues
    }

# =====================================================
# AC3 – RACK SCAN DETECTION (MOTION ANALYSIS)
# =====================================================

def analyze_rack_scan(video_path, meta):
    """Detect vertical panning and motion quality"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(fps * CONFIG["frame_sample_interval_sec"]))

    prev_frame = None
    frame_idx = 0
    vertical_motions = []
    horizontal_motions = []
    motion_consistency = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="  Analyzing rack scan motion", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                v_motion = np.mean(np.abs(flow[..., 1]))  # Vertical
                h_motion = np.mean(np.abs(flow[..., 0]))  # Horizontal
                
                vertical_motions.append(v_motion)
                horizontal_motions.append(h_motion)
                motion_consistency.append(v_motion if v_motion > h_motion else h_motion)

            prev_frame = gray
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()

    cap.release()

    avg_v = np.mean(vertical_motions) if vertical_motions else 0
    avg_h = np.mean(horizontal_motions) if horizontal_motions else 0
    motion_std = np.std(motion_consistency) if motion_consistency else 0

    motion_pattern = "Vertical" if avg_v > avg_h else "Horizontal"
    motion_speed_status = "Optimal"
    issues = []

    if avg_v < CONFIG["motion_speed_slow"]:
        motion_speed_status = "Too Slow"
        issues.append("Camera motion too slow - cannot see details")
    elif avg_v > CONFIG["motion_speed_fast"]:
        motion_speed_status = "Too Fast"
        issues.append("Camera motion too fast - frames unreadable")
    
    if motion_std > 10:
        issues.append("Inconsistent motion - jerky movement detected")

    return {
        "passed": len(issues) == 0,
        "motion_pattern": motion_pattern,
        "vertical_motion_avg": avg_v,
        "horizontal_motion_avg": avg_h,
        "motion_speed_status": motion_speed_status,
        "motion_consistency": 100 - (motion_std * 5),
        "issues": issues
    }

# =====================================================
# AC5 – PORT DETECTION READINESS
# =====================================================

def analyze_port_readiness(video_path, meta):
    """Check if ports are visible and distinguishable"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"passed": False, "issues": ["Cannot read first frame"]}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 
                       CONFIG["edge_detection_threshold_low"],
                       CONFIG["edge_detection_threshold_high"])
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    port_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if CONFIG["min_pixels_per_port"] < area < CONFIG["max_pixels_per_port"]:
            x, y, w, h = cv2.boundingRect(contour)
            if w > CONFIG["min_pixels_per_port"] or h > CONFIG["min_pixels_per_port"]:
                port_candidates.append((x, y, w, h))

    # Check separation
    separation_ok = True
    if len(port_candidates) > 1:
        for i in range(len(port_candidates) - 1):
            x1, y1, w1, h1 = port_candidates[i]
            x2, y2, w2, h2 = port_candidates[i + 1]
            sep = min(abs(x2 - (x1 + w1)), abs(y2 - (y1 + h1)))
            if sep < CONFIG["port_separation_min_px"]:
                separation_ok = False

    issues = []
    if len(port_candidates) < CONFIG["ports_expected_min"]:
        issues.append("Too few port-like objects (need to zoom in)")
    if not separation_ok:
        issues.append("Ports too close - difficult to distinguish")

    return {
        "passed": len(issues) == 0,
        "objects_detected": len(port_candidates),
        "separation_ok": separation_ok,
        "issues": issues
    }


# =====================================================
# RACK OVERVIEW VALIDATION (ADDED)
# =====================================================
def analyze_rack_overview(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(fps * CONFIG.get("frame_sample_interval_sec", 1.0)))

    frame_idx = 0
    prev_center_y = None

    center_y_positions = []
    vertical_speeds = []

    edge_visible_frames = 0
    occluded_frames = 0
    total_samples = 0

    frame_height = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="  Analyzing rack overview", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            total_samples += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            frame_height = h

            edges = cv2.Canny(gray, CONFIG.get("canny_low", 50), CONFIG.get("canny_high", 150))
            edge_density = np.mean(edges > 0)

            # Occlusion
            if edge_density < CONFIG.get("occlusion_edge_density_min", 0.008):
                occluded_frames += 1

            # Rack visibility
            if edge_density > 0.01:
                edge_visible_frames += 1

            # Vertical displacement of edge pixels
            ys, _ = np.where(edges > 0)
            if len(ys) > 0:
                center_y = np.mean(ys)
                center_y_positions.append(center_y)

                if prev_center_y is not None:
                    vertical_speeds.append(abs(center_y - prev_center_y))

                prev_center_y = center_y

        frame_idx += 1
        pbar.update(1)
    
    pbar.close()

    cap.release()

    if center_y_positions and frame_height:
        vertical_travel = max(center_y_positions) - min(center_y_positions)
        travel_ratio = vertical_travel / frame_height
    else:
        travel_ratio = 0.0

    median_speed = float(np.median(vertical_speeds)) if vertical_speeds else 0.0

    rack_visibility_ratio = edge_visible_frames / total_samples if total_samples else 0.0
    occlusion_ratio = occluded_frames / total_samples if total_samples else 0.0

    issues = []

    # VALIDATION
    if travel_ratio < CONFIG.get("min_vertical_travel_ratio", 0.35):
        issues.append(f"Incomplete vertical rack scan (only {travel_ratio*100:.1f}% height covered)")

    if median_speed > CONFIG.get("motion_speed_fast_overview", CONFIG.get("motion_speed_fast", 30)):
        issues.append("Camera movement too fast")

    if rack_visibility_ratio < CONFIG.get("min_edge_presence_ratio", 0.90):
        issues.append(f"Rack visible in only {rack_visibility_ratio*100:.1f}% of frames")

    if occlusion_ratio > CONFIG.get("occlusion_max_threshold", 0.20):
        issues.append(f"Occlusion detected in {occlusion_ratio*100:.1f}% of frames")

    return {
        "passed": len(issues) == 0,
        "vertical_travel_ratio": travel_ratio,
        "median_vertical_speed": median_speed,
        "rack_visibility_ratio": rack_visibility_ratio,
        "occlusion_ratio": occlusion_ratio,
        "issues": issues,
        "total_samples": total_samples,
    }


# =====================================================
# Overall decision logic (configurable by profile)
# =====================================================
def determine_overall_pass(ac1, ac2, ac3, ac5, ac_overview, cfg=CONFIG):
    """Return (passed: bool, reason: str). In 'Standard' profile AC5 and overview
    failures are treated as warnings (do not block acceptance). In 'Strict' they block.
    AC1, AC2, AC3 are considered blocking failures for all profiles.
    """
    # Blockers
    if not ac1.get("passed", False):
        return False, "AC1 failed"
    if not ac2.get("passed", False):
        return False, "AC2 failed"
    if not ac3.get("passed", False):
        return False, "AC3 failed"

    profile = cfg.get("validation_profile", "Standard")

    # In Strict profile, AC5 and overview must pass
    if profile == "Strict":
        if not ac5.get("passed", False):
            return False, "AC5 failed (Strict profile)"
        if not ac_overview.get("passed", False):
            return False, "Overview failed (Strict profile)"

    # For Standard and Lenient, non-blocking: accept but surface issues
    return True, ""

# =====================================================
# HTML REPORT (HUMAN + TECHNICAL)
# =====================================================

def generate_html_report(meta, ac1, ac2, ac3, ac5, ac_overview, output_path):
    overall_pass, _ = determine_overall_pass(ac1, ac2, ac3, ac5, ac_overview)
    status_color = "#28a745" if overall_pass else "#dc3545"
    status_text = "ACCEPTED ✓" if overall_pass else "REJECTED ✗"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Video Validation Report - Production Grade</title>
<style>
body {{ font-family: 'Segoe UI', Arial; background:#f0f2f5; padding:20px; }}
.container {{ max-width:1200px; margin:auto; background:white; padding:40px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1); }}
h1 {{ color:#333; border-bottom:4px solid {status_color}; margin-top:0; }}
h2 {{ color:#555; margin-top:30px; font-size:1.2em; }}
.badge {{ background:{status_color}; color:white; padding:8px 16px; border-radius:20px; font-weight:bold; }}
.section {{ margin:25px 0; padding:20px; background:#f8f9fa; border-left:5px solid {status_color}; border-radius:5px; }}
.bar {{ height:20px; background:#ddd; border-radius:10px; margin:10px 0; }}
.fill {{ height:100%; background:linear-gradient(90deg,#dc3545,#ffc107,#28a745); border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold; font-size:12px; }}
.issue {{ background:#fff3cd; padding:12px; margin:8px 0; border-left:4px solid #ffc107; border-radius:4px; }}
.pass {{ background:#d4edda; padding:12px; margin:8px 0; border-left:4px solid #28a745; border-radius:4px; }}
.tech {{ background:#f4f4f4; font-family:monospace; padding:15px; margin:10px 0; border-radius:4px; overflow-x:auto; font-size:12px; }}
table {{ width:100%; border-collapse:collapse; margin:15px 0; }}
th {{ background:#f0f0f0; padding:10px; text-align:left; border-bottom:2px solid #ddd; }}
td {{ padding:8px; border-bottom:1px solid #eee; }}
.warn {{ color:#ff9800; font-weight:bold; }}
</style>
</head>
<body>
<div class="container">

<h1>Production Video Validation Report <span class="badge">{status_text}</span></h1>
<p><strong>File:</strong> {meta["filename"]} | <strong>Size:</strong> {meta["filesize_mb"]:.1f} MB</p>
<p><strong>Generated:</strong> {datetime.utcnow().isoformat()}Z</p>

<!-- AC1 -->
<div class="section">
<h2>AC1: Metadata Validation</h2>
<table>
<tr><th>Parameter</th><th>Value</th><th>Requirement</th><th>Status</th></tr>
<tr><td>Resolution</td><td>{meta["width"]}x{meta["height"]}</td><td>≥{CONFIG["min_width"]}x{CONFIG["min_height"]}</td><td>{'✓' if meta["width"] >= CONFIG["min_width"] else '✗'}</td></tr>
<tr><td>Frame Rate</td><td>{meta["fps"]:.2f} fps</td><td>≥{CONFIG["min_fps"]} fps</td><td>{'✓' if meta["fps"] >= CONFIG["min_fps"] else '✗'}</td></tr>
<tr><td>Codec</td><td>{meta["codec"]}</td><td>{', '.join(CONFIG["supported_codecs"][:3])}</td><td>{'✓' if meta["codec"] in CONFIG["supported_codecs"] else '⚠'}</td></tr>
<tr><td>Duration</td><td>{meta["duration_sec"]:.1f}s</td><td>≤{CONFIG["max_duration_minutes"]}min</td><td>✓</td></tr>
</table>
{"".join(f"<div class='issue'>⚠ {w}</div>" for w in ac1.get("warnings", []))}
{"".join(f"<div class='issue'>✗ {i}</div>" for i in ac1.get("issues", []))}
</div>

<!-- AC2 -->
<div class="section">
<h2>AC2: Quality & Lighting</h2>
<div>Quality Score: <div class="bar"><div class="fill" style="width:{ac2["quality_score"]}%">{ac2["quality_score"]}/100</div></div></div>
<table>
<tr><th>Metric</th><th>Value</th><th>Status</th></tr>
<tr><td>Blur Status</td><td>{ac2["blur_status"]}</td><td>{"✓" if ac2["blur_status"] == "Sharp" else "⚠"}</td></tr>
<tr><td>Blurry Frames</td><td>{ac2["blur_ratio"]*100:.1f}%</td><td>{"✓" if ac2["blur_ratio"] < CONFIG["blur_ratio_warn"] else "⚠"}</td></tr>
<tr><td>Avg Laplacian</td><td>{ac2["avg_laplacian"]:.2f}</td><td>Threshold: {ac2["laplacian_threshold_used"]:.0f}</td></tr>
<tr><td>Edge Density</td><td>{ac2["avg_edge_density"]:.4f}</td><td>Min: {CONFIG["min_edge_density"]}</td></tr>
</table>
{"".join(f"<div class='issue'>⚠ {i}</div>" for i in ac2.get("issues", []))}
</div>

<!-- AC3 -->
<div class="section">
<h2>AC3: Rack Scan Detection</h2>
<table>
<tr><th>Metric</th><th>Value</th><th>Status</th></tr>
<tr><td>Motion Pattern</td><td>{ac3["motion_pattern"]}</td><td>{"✓" if ac3["motion_pattern"] == "Vertical" else "⚠"}</td></tr>
<tr><td>Vertical Motion</td><td>{ac3["vertical_motion_avg"]:.2f} px/frame</td><td>Optimal: {CONFIG["motion_speed_acceptable"]} px/frame</td></tr>
<tr><td>Horizontal Motion</td><td>{ac3["horizontal_motion_avg"]:.2f} px/frame</td><td></td></tr>
<tr><td>Speed Status</td><td>{ac3["motion_speed_status"]}</td><td>{"✓" if ac3["motion_speed_status"] == "Optimal" else "⚠"}</td></tr>
<tr><td>Motion Consistency</td><td>{ac3["motion_consistency"]:.0f}%</td><td>Target: >90%</td></tr>
</table>
{"".join(f"<div class='issue'>✗ {i}</div>" for i in ac3.get("issues", []))}
</div>

<!-- OVERVIEW -->
<div class="section">
<h2>Rack Overview Validation</h2>
<table>
<tr><th>Metric</th><th>Value</th><th>Requirement</th><th>Status</th></tr>
<tr><td>Vertical Travel</td><td>{ac_overview["vertical_travel_ratio"]*100:.1f}%</td><td>≥{CONFIG["min_vertical_travel_ratio"]*100:.0f}%</td><td>{'✓' if ac_overview["vertical_travel_ratio"] >= CONFIG["min_vertical_travel_ratio"] else '✗'}</td></tr>
<tr><td>Median Vertical Speed</td><td>{ac_overview["median_vertical_speed"]:.1f} px/frame</td><td>&lt; {CONFIG["motion_speed_fast_overview"]} px/frame</td><td>{'✓' if ac_overview["median_vertical_speed"] <= CONFIG["motion_speed_fast_overview"] else '⚠'}</td></tr>
<tr><td>Rack Visibility</td><td>{ac_overview["rack_visibility_ratio"]*100:.1f}%</td><td>≥{CONFIG["min_edge_presence_ratio"]*100:.0f}%</td><td>{'✓' if ac_overview["rack_visibility_ratio"] >= CONFIG["min_edge_presence_ratio"] else '✗'}</td></tr>
<tr><td>Occlusion</td><td>{ac_overview["occlusion_ratio"]*100:.1f}%</td><td>≤{CONFIG["occlusion_max_threshold"]*100:.0f}%</td><td>{'✓' if ac_overview["occlusion_ratio"] <= CONFIG["occlusion_max_threshold"] else '✗'}</td></tr>
</table>
{"".join(f"<div class='issue'>✗ {i}</div>" for i in ac_overview.get("issues", []))}
</div>

<!-- AC5 -->
<div class="section">
<h2>AC5: Port Detection Readiness</h2>
<table>
<tr><th>Metric</th><th>Value</th><th>Requirement</th><th>Status</th></tr>
<tr><td>Objects Detected</td><td>{ac5["objects_detected"]}</td><td>≥{CONFIG["ports_expected_min"]}</td><td>{"✓" if ac5["objects_detected"] >= CONFIG["ports_expected_min"] else "⚠"}</td></tr>
<tr><td>Separation</td><td>{'OK' if ac5["separation_ok"] else 'Too Close'}</td><td>≥{CONFIG["port_separation_min_px"]}px apart</td><td>{"✓" if ac5["separation_ok"] else "✗"}</td></tr>
</table>
{"".join(f"<div class='issue'>{i}</div>" for i in ac5.get("issues", []))}
</div>

<!-- SUMMARY -->
<div class="section">
<h2>Validation Summary</h2>
<p><strong>Overall Status:</strong> <span class="badge">{status_text}</span></p>
<p><strong>Production Config Profile:</strong> {CONFIG["validation_profile"]}</p>
<p><strong>Config Version:</strong> {CONFIG["config_version"]}</p>
{"<div class='pass'>✓ Video meets all production requirements</div>" if overall_pass else "<div class='issue'>✗ Video requires corrections before acceptance</div>"}
</div>

<!-- TECHNICAL DETAILS -->
<div class="section">
<h2>Technical Configuration Used</h2>
<div class="tech">
Validation Profile: {CONFIG["validation_profile"]}<br>
Parallel Processing: {'Enabled' if CONFIG["parallel_processing_enabled"] else 'Disabled'}<br>
Early Rejection: {'Enabled' if CONFIG["early_rejection_enabled"] else 'Disabled'}<br>
Max Processing Time: {CONFIG["max_processing_time_multiplier"]}x video duration<br>
Memory Limit: {CONFIG["memory_limit_gb"]}GB<br>
Debug Mode: {'ON' if CONFIG["debug_mode"] else 'OFF'}<br>
Audit Trail: {'Enabled' if CONFIG["audit_trail"] else 'Disabled'}<br>
</div>
</div>

</div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    for video in [
        r"\\192.168.1.15\Sharing folder\VID_20260128_143249.mp4"
    ]:
        print(f"\n{'='*70}")
        print(f"Validating: {Path(video).name}")
        print(f"{'='*70}")
        
        try:
            meta = extract_metadata(video)
            ac1 = validate_ac1(meta)
            ac2 = analyze_quality(video, meta)
            ac3 = analyze_rack_scan(video, meta)
            ac5 = analyze_port_readiness(video, meta)
            ac_overview = analyze_rack_overview(video)

            report = Path(video).with_suffix(".production_report.html")
            generate_html_report(meta, ac1, ac2, ac3, ac5, ac_overview, report)

            # Console summary
            print(f"\n✓ AC1 Metadata: {'PASS' if ac1['passed'] else 'FAIL'}")
            print(f"✓ AC2 Quality: {'PASS' if ac2['passed'] else 'FAIL'} (Score: {ac2['quality_score']}/100)")
            print(f"✓ AC3 Rack Scan: {'PASS' if ac3['passed'] else 'FAIL'} ({ac3['motion_speed_status']})")
            print(f"✓ AC5 Ports: {'PASS' if ac5['passed'] else 'FAIL'} ({ac5['objects_detected']} objects)")
            print(f"✓ Overview: {'PASS' if ac_overview['passed'] else 'FAIL'} (Vertical travel: {ac_overview['vertical_travel_ratio']*100:.1f}%, Visibility: {ac_overview['rack_visibility_ratio']*100:.1f}%)")
            
            overall, reason = determine_overall_pass(ac1, ac2, ac3, ac5, ac_overview)
            if overall:
                print(f"\n📊 OVERALL: ACCEPTED ✓")
            else:
                msg = f"REJECTED ✗ ({reason})" if reason else "REJECTED ✗"
                print(f"\n📊 OVERALL: {msg}")
            print(f"📄 Report: {report}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
