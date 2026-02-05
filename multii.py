#!/usr/bin/env python3
"""
Multi-Image Rack Processing Pipeline
- Processes multiple rack images
- Identifies rack units in each image
- Detects switches, patch panels, cables, and ports within each unit
- Groups similar devices across all images
- Saves only best representative from each group
- Outputs in unit-based folder structure: output/units/unit1/patchpanel/, output/units/unit1/ports/, etc.
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps
import imagehash
import torch

# ============================
# Configuration
# ============================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("[WARN] Running on CPU - Processing will be slower. Consider using GPU.")

# Default paths (can be overridden by command-line args)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "files", "multiple-images")
OUTPUT_ROOT = os.path.join(BASE_DIR, "segmented_output")

MODEL_PATHS = {
    "rack": os.path.join(BASE_DIR, "Trained_Models", "rack_best.pt"),
    "rack_unit": os.path.join(BASE_DIR, "Trained_Models", "rack_unit.pt"),
    "switch": os.path.join(BASE_DIR, "Trained_Models", "switch_patch.pt"),
    "cables": os.path.join(BASE_DIR, "Trained_Models", "cable_best.pt"),
    "port": os.path.join(BASE_DIR, "Trained_Models", "port_count.pt"),
}

# Detection thresholds
CONF_THRESHOLDS = {
    "rack": 0.6,
    "rack_unit": 0.30,
    "switch": 0.44,
    "cables": 0.18,
    "port": 0.2,
}

# Grouping parameters
HASH_BITS = 32
ROTATION_STEP = 10
ORB_N_FEATURES = 500
ORB_DISTANCE_LIMIT = 40
SSIM_SIZE = 256
MIN_GROUP_SCORE = 0.5
MIN_GROUP_SCORE_PATCH = 0.6

# ============================
# Utility Functions
# ============================

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def list_images(root: str) -> List[str]:
    """Find all image files in directory."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, f"**/*{e}"), recursive=True))
    return sorted(files)

def read_bgr(path: str) -> Optional[np.ndarray]:
    """Read image as BGR array."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img if img is not None else None
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def crop_with_margin(bgr, box, margin=10, min_dim=128):
    """Crop image region with margin and minimum dimension."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W - 1, x2 + margin)
    y2 = min(H - 1, y2 + margin)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop = bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    
    if ch < min_dim or cw < min_dim:
        crop = cv2.resize(crop, (max(cw, min_dim), max(ch, min_dim)), 
                         interpolation=cv2.INTER_CUBIC)
    return crop

def save_image(img_bgr, path: str):
    """Save image to file."""
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# ============================
# Model Loading
# ============================

def load_yolo(model_path: str):
    """Load YOLO model."""
    if not os.path.exists(model_path):
        print(f"[WARN] Model not found: {model_path}")
        return None
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        if DEVICE == "cuda:0":
            model.to(DEVICE)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        return None

def load_models(model_paths: Dict[str, str]) -> Dict[str, any]:
    """Load all YOLO models."""
    models = {}
    for name, path in model_paths.items():
        print(f"Loading {name} model...")
        model = load_yolo(path)
        if model:
            models[name] = model
    return models

# ============================
# Detection Functions
# ============================

def get_class_ids(model, class_names: set) -> List[int]:
    """Get class IDs matching given names."""
    if model is None:
        return []
    
    names = model.names
    if isinstance(names, dict):
        names_dict = {int(k): str(v) for k, v in names.items()}
    else:
        names_dict = {i: str(n) for i, n in enumerate(names)}
    
    class_names_lower = {n.lower() for n in class_names}
    return [cid for cid, name in names_dict.items() if name.lower() in class_names_lower]

def detect_racks(img_bgr: np.ndarray, rack_model) -> List[Tuple[list, float]]:
    """Detect rack regions in image."""
    if rack_model is None:
        return []
    
    conf = CONF_THRESHOLDS.get("rack", 0.6)
    results = rack_model(img_bgr, conf=conf, verbose=False)
    
    detections = []
    for box, conf_val in zip(results[0].boxes.xyxy, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        detections.append(([x1, y1, x2, y2], float(conf_val)))
    
    return detections

def detect_units(img_bgr: np.ndarray, rack_unit_model) -> List[Tuple[list, float]]:
    """Detect rack units within image."""
    if rack_unit_model is None:
        return []
    
    conf = CONF_THRESHOLDS.get("rack_unit", 0.30)
    results = rack_unit_model(img_bgr, conf=conf, verbose=False)
    
    detections = []
    for box, conf_val in zip(results[0].boxes.xyxy, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        detections.append(([x1, y1, x2, y2], float(conf_val)))
    
    return sorted(detections, key=lambda x: x[0][1])  # Sort by Y coordinate

def detect_devices(img_bgr: np.ndarray, device_model, device_type: str) -> List[Tuple[list, str, float]]:
    """Detect devices (switches/patch panels) in image."""
    if device_model is None:
        return []
    
    conf = CONF_THRESHOLDS.get("switch", 0.44)
    results = device_model(img_bgr, conf=conf, verbose=False)
    
    detections = []
    switch_ids = get_class_ids(device_model, {"switch", "network_switch"})
    patch_ids = get_class_ids(device_model, {"patch", "patchpanel", "patch_panel"})
    
    for box, cls_id, conf_val in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        
        # Classify as switch or patch panel
        if cls_id in switch_ids:
            device_class = "switch"
        elif cls_id in patch_ids:
            device_class = "patch_panel"
        else:
            continue
        
        if device_type == device_class:
            detections.append(([x1, y1, x2, y2], device_class, float(conf_val)))
    
    return detections

def detect_ports(img_bgr: np.ndarray, port_model) -> List[Tuple[list, str, float]]:
    """Detect ports (connected/empty) in image."""
    if port_model is None:
        return []
    
    conf = CONF_THRESHOLDS.get("port", 0.2)
    results = port_model(img_bgr, conf=conf, verbose=False)
    
    detections = []
    connected_ids = get_class_ids(port_model, {"connected_port", "connected"})
    empty_ids = get_class_ids(port_model, {"empty_port", "empty"})
    
    for box, cls_id, conf_val in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        
        if cls_id in connected_ids:
            port_type = "connected_port"
        elif cls_id in empty_ids:
            port_type = "empty_port"
        else:
            continue
        
        detections.append(([x1, y1, x2, y2], port_type, float(conf_val)))
    
    return detections

def detect_cables(img_bgr: np.ndarray, cable_model) -> List[Tuple[list, float]]:
    """Detect cables in image."""
    if cable_model is None:
        return []
    
    conf = CONF_THRESHOLDS.get("cables", 0.18)
    results = cable_model(img_bgr, conf=conf, verbose=False)
    
    detections = []
    for box, conf_val in zip(results[0].boxes.xyxy, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        detections.append(([x1, y1, x2, y2], float(conf_val)))
    
    return detections

# ============================
# Grouping Functions
# ============================

class DeviceGroup:
    """Represents a group of similar devices across images."""
    
    def __init__(self, gid: int):
        self.gid = f"group_{gid:04d}"
        self.members: List[Dict] = []  # {src_path, crop_bgr, unit_id, image_name}
        self.rep_pil = None
        self.rep_hash = None
    
    def add_member(self, src_path: str, crop_bgr: np.ndarray, unit_id: str, image_name: str):
        """Add a member to this group."""
        self.members.append({
            "src_path": src_path,
            "crop_bgr": crop_bgr,
            "unit_id": unit_id,
            "image_name": image_name
        })
    
    def set_representative(self, pil_img: Image.Image):
        """Set the representative image for this group."""
        self.rep_pil = pil_img.copy()
        self.rep_hash = imagehash.phash(pil_img)
    
    def similarity_to_image(self, pil_img: Image.Image) -> float:
        """Calculate similarity score to an image."""
        if self.rep_hash is None:
            return 0.0
        
        img_hash = imagehash.phash(pil_img)
        distance = (self.rep_hash - img_hash)
        similarity = 1.0 - (distance / HASH_BITS)
        return max(0.0, similarity)
    
    def choose_best_member(self) -> Optional[np.ndarray]:
        """Select best representative crop from members."""
        if not self.members:
            return None
        
        best_idx = 0
        best_score = -1.0
        
        for i, member in enumerate(self.members):
            crop = member["crop_bgr"]
            h, w = crop.shape[:2]
            
            # Score based on size and sharpness
            size_score = (w * h) / 1e6
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharp_score = min(1.0, sharp / 1000.0)
            
            score = 0.6 * size_score + 0.4 * sharp_score
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        return self.members[best_idx]["crop_bgr"]

def group_devices(all_devices: List[Dict], min_score: float = MIN_GROUP_SCORE) -> List[DeviceGroup]:
    """Group similar devices across all images."""
    groups: List[DeviceGroup] = []
    
    for idx, device in enumerate(all_devices):
        crop_bgr = device["crop_bgr"]
        pil_img = bgr_to_pil(crop_bgr)
        
        # Try to find best matching group
        best_group_idx = -1
        best_similarity = -1.0
        
        for g_idx, group in enumerate(groups):
            sim = group.similarity_to_image(pil_img)
            if sim > best_similarity:
                best_similarity = sim
                best_group_idx = g_idx
        
        # Add to existing group or create new one
        if best_group_idx >= 0 and best_similarity >= min_score:
            groups[best_group_idx].add_member(
                device["src_path"],
                crop_bgr,
                device.get("unit_id", "unknown"),
                device.get("image_name", "unknown")
            )
        else:
            # Create new group
            new_group = DeviceGroup(len(groups) + 1)
            new_group.add_member(
                device["src_path"],
                crop_bgr,
                device.get("unit_id", "unknown"),
                device.get("image_name", "unknown")
            )
            new_group.set_representative(pil_img)
            groups.append(new_group)
    
    return groups

# ============================
# Main Processing
# ============================

def process_multi_images(input_dir: str, output_root: str, models: Dict[str, any]):
    """Process multiple images and save organized output."""
    
    # Clean output directory before processing (avoid stale results)
    if os.path.exists(output_root):
        print(f"[INFO] Cleaning output directory: {output_root}")
        shutil.rmtree(output_root)
    
    ensure_dir(output_root)
    
    # Find all images
    images = list_images(input_dir)
    if not images:
        print(f"[ERROR] No images found in {input_dir}")
        return
    
    print(f"[OK] Found {len(images)} images to process")
    
    # Storage for grouping across all images
    all_switches: List[Dict] = []
    all_patches: List[Dict] = []
    
    # Storage for hierarchical output (by unit)
    unit_data = {}  # unit_id -> {switches, patches, ports}
    
    # Process each image
    for img_idx, img_path in enumerate(images, 1):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[{img_idx}/{len(images)}] Processing: {img_name}")
        
        bgr = read_bgr(img_path)
        if bgr is None:
            print(f"[WARN] Skipped: Failed to read image")
            continue
        
        H, W = bgr.shape[:2]
        
        # Detect rack
        rack_detections = detect_racks(bgr, models.get("rack"))
        if not rack_detections:
            print(f"[WARN] No rack detected in this image")
            continue
        
        # Use the best rack detection
        rack_box, _ = rack_detections[0]
        x1, y1, x2, y2 = rack_box
        rack_crop = bgr[y1:y2, x1:x2]
        
        # Detect units within the rack
        unit_detections = detect_units(rack_crop, models.get("rack_unit"))
        if not unit_detections:
            print(f"[WARN] No units detected in this rack")
            continue
        
        print(f"[OK] Found {len(unit_detections)} units")
        
        # Process each unit
        for unit_idx, (unit_box, unit_conf) in enumerate(unit_detections, 1):
            unit_id = f"unit{unit_idx}"
            ux1, uy1, ux2, uy2 = unit_box
            unit_crop = rack_crop[uy1:uy2, ux1:ux2]
            
            # Initialize unit data if not exists
            if unit_id not in unit_data:
                unit_data[unit_id] = {
                    "switches": [],
                    "patch_panels": [],
                    "connected_port": [],
                    "empty_port": [],
                    "cables": []
                }
            
            # Detect switches in this unit
            switches = detect_devices(unit_crop, models.get("switch"), "switch")
            for sw_box, sw_type, sw_conf in switches:
                crop = crop_with_margin(unit_crop, sw_box)
                if crop is not None:
                    device_data = {
                        "src_path": img_path,
                        "crop_bgr": crop,
                        "unit_id": unit_id,
                        "image_name": img_name,
                        "conf": sw_conf
                    }
                    all_switches.append(device_data)
                    unit_data[unit_id]["switches"].append(device_data)
            
            # Detect patch panels in this unit
            patches = detect_devices(unit_crop, models.get("switch"), "patch_panel")
            for pp_box, pp_type, pp_conf in patches:
                crop = crop_with_margin(unit_crop, pp_box)
                if crop is not None:
                    device_data = {
                        "src_path": img_path,
                        "crop_bgr": crop,
                        "unit_id": unit_id,
                        "image_name": img_name,
                        "conf": pp_conf
                    }
                    all_patches.append(device_data)
                    unit_data[unit_id]["patch_panels"].append(device_data)
            
            # Detect ports in this unit
            ports = detect_ports(unit_crop, models.get("port"))
            for port_box, port_type, port_conf in ports:
                crop = crop_with_margin(unit_crop, port_box)
                if crop is not None:
                    port_data = {
                        "type": port_type,
                        "crop_bgr": crop,
                        "unit_id": unit_id,
                        "image_name": img_name
                    }
                    # Ensure port_type key exists before appending
                    if port_type not in unit_data[unit_id]:
                        unit_data[unit_id][port_type] = []
                    unit_data[unit_id][port_type].append(port_data)
            
            # Detect cables in this unit
            cables = detect_cables(unit_crop, models.get("cables"))
            for cable_box, cable_conf in cables:
                crop = crop_with_margin(unit_crop, cable_box)
                if crop is not None:
                    cable_data = {
                        "crop_bgr": crop,
                        "unit_id": unit_id,
                        "image_name": img_name,
                        "confidence": cable_conf
                    }
                    unit_data[unit_id]["cables"].append(cable_data)
    
    # Group switches and patch panels across all units
    print(f"\n{'='*60}")
    print(f"Grouping {len(all_switches)} switches...")
    switch_groups = group_devices(all_switches, MIN_GROUP_SCORE)
    print(f"[OK] Created {len(switch_groups)} switch groups")
    
    print(f"Grouping {len(all_patches)} patch panels...")
    patch_groups = group_devices(all_patches, MIN_GROUP_SCORE_PATCH)
    print(f"[OK] Created {len(patch_groups)} patch panel groups")
    
    # Save organized output by unit
    print(f"\n{'='*60}")
    print(f"Saving organized outputs...")
    
    # Save rack images to root output directory
    rack_dir = os.path.join(output_root, "rack")
    ensure_dir(rack_dir)
    
    rack_count = 0
    for img_idx, img_path in enumerate(images, 1):
        bgr = read_bgr(img_path)
        if bgr is None:
            continue
        
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Detect racks in full image using rack model
        racks = detect_racks(bgr, models.get("rack"))
        for rack_box, rack_conf in racks:
            crop = crop_with_margin(bgr, rack_box)
            if crop is not None:
                rack_count += 1
                save_path = os.path.join(rack_dir, f"rack_{rack_count:03d}.jpg")
                save_image(crop, save_path)
    
    print(f"[OK] Saved {rack_count} rack images to: {rack_dir}")
    
    # Save cables to a separate root folder (not inside unit folders)
    cables_root = os.path.join(output_root, "cables")
    ensure_dir(cables_root)
    cable_count = 0
    for unit_id, unit_info in unit_data.items():
        for c_idx, cable_data in enumerate(unit_info.get("cables", []), 1):
            crop = cable_data.get("crop_bgr")
            if crop is None:
                continue
            cable_count += 1
            # include unit id and original image name for traceability
            img_name = cable_data.get("image_name") or f"img{cable_count:03d}"
            save_name = f"cable_{unit_id}_{img_name}_{c_idx:03d}.jpg"
            save_path = os.path.join(cables_root, save_name)
            save_image(crop, save_path)

    print(f"[OK] Saved {cable_count} cables to: {cables_root}")

    # Save units with mutually exclusive device types (switch OR patchpanel, not both)
    units_dir = os.path.join(output_root, "units")
    ensure_dir(units_dir)
    
    for unit_id, unit_info in unit_data.items():
        # Determine presence of devices/ports in this unit
        has_switches = len(unit_info.get("switches", [])) > 0
        has_patches = len(unit_info.get("patch_panels", [])) > 0
        has_ports = len(unit_info.get("connected_port", [])) > 0 or len(unit_info.get("empty_port", [])) > 0

        # If a device (switch/patch) was detected but NO ports were found, skip storing/displaying per user request
        if (has_switches or has_patches) and not has_ports:
            print(f"[INFO] Skipping unit {unit_id} - device detected but no ports present; not storing or displaying")
            continue

        # Skip units with unknown device types (only process units with valid devices)
        if not (has_switches or has_patches):
            if has_ports:
                num_ports = len(unit_info.get("connected_port", [])) + len(unit_info.get("empty_port", []))
                print(f"[INFO] Skipping unit {unit_id} - no valid device found ({num_ports} ports detected, 0 devices)")
            continue
        
        # Log what devices were detected for debugging
        num_switches = len(unit_info.get("switches", []))
        num_patches = len(unit_info.get("patch_panels", []))
        num_ports = len(unit_info.get("connected_port", [])) + len(unit_info.get("empty_port", []))
        print(f"[DEBUG] Unit {unit_id}: switches={num_switches}, patches={num_patches}, ports={num_ports}")

        # Only create unit folder when there is a switch or patch panel
        unit_dir = os.path.join(units_dir, unit_id)

        # Track saved representative hashes to avoid duplicates (local to unit)
        saved_switch_hashes = []
        saved_patch_hashes = []

        # Save switches if present (only if no patch panels) - select ONE BEST per unit
        if has_switches and not has_patches:
            ensure_dir(unit_dir)
            switches_dir = os.path.join(unit_dir, "switch")
            ensure_dir(switches_dir)
            
            # Collect all switch members in this unit from ALL groups
            all_unit_switches = []
            for group in switch_groups:
                unit_members = [m for m in group.members if m["unit_id"] == unit_id]
                all_unit_switches.extend(unit_members)
            
            if all_unit_switches:
                # Find the SINGLE BEST switch among all members in this unit
                best_crop = None
                best_score = -1.0
                
                for member in all_unit_switches:
                    crop = member["crop_bgr"]
                    h, w = crop.shape[:2]
                    size_score = (w * h) / 1e6
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                    sharp_score = min(1.0, sharp / 1000.0)
                    score = 0.6 * size_score + 0.4 * sharp_score
                    
                    if score > best_score:
                        best_score = score
                        best_crop = crop
                
                if best_crop is not None:
                    save_path = os.path.join(switches_dir, "switch_best.jpg")
                    save_image(best_crop, save_path)
                    print(f"[OK] Saved best switch for unit {unit_id} (score: {best_score:.3f})")

        
        # Save patch panels only if no switches (mutually exclusive) - select ONE BEST per unit
        elif has_patches and not has_switches:
            ensure_dir(unit_dir)
            patches_dir = os.path.join(unit_dir, "patchpanel")
            ensure_dir(patches_dir)
            
            # Collect all patch panel members in this unit from ALL groups
            all_unit_patches = []
            for group in patch_groups:
                unit_members = [m for m in group.members if m["unit_id"] == unit_id]
                all_unit_patches.extend(unit_members)
            
            if all_unit_patches:
                # Find the SINGLE BEST patch panel among all members in this unit
                best_crop = None
                best_score = -1.0
                
                for member in all_unit_patches:
                    crop = member["crop_bgr"]
                    h, w = crop.shape[:2]
                    size_score = (w * h) / 1e6
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                    sharp_score = min(1.0, sharp / 1000.0)
                    score = 0.6 * size_score + 0.4 * sharp_score
                    
                    if score > best_score:
                        best_score = score
                        best_crop = crop
                
                if best_crop is not None:
                    save_path = os.path.join(patches_dir, "patchpanel_best.jpg")
                    save_image(best_crop, save_path)
                    print(f"[OK] Saved best patch panel for unit {unit_id} (score: {best_score:.3f})")

        
        # Save ports only if the unit has a device (switch or patch panel)
        total_ports_saved = 0
        if has_ports and (has_switches or has_patches):
            connected_root = os.path.join(unit_dir, "ports", "connected")
            empty_root = os.path.join(unit_dir, "ports", "empty")
            ensure_dir(connected_root)
            ensure_dir(empty_root)

            for port_type in ["connected_port", "empty_port"]:
                ports_list = unit_info.get(port_type, [])
                if not ports_list:
                    continue

                for p_idx, port_data in enumerate(ports_list, 1):
                    port_type_short = "connected" if "connected" in port_type else "empty"
                    save_dir = connected_root if port_type_short == "connected" else empty_root
                    save_path = os.path.join(save_dir, f"{unit_id}_port_{port_type_short}_{p_idx:03d}.jpg")
                    try:
                        ok = cv2.imwrite(save_path, port_data["crop_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        if not ok:
                            print(f"[WARN] Failed to write port image: {save_path}")
                        else:
                            total_ports_saved += 1
                    except Exception as e:
                        print(f"[WARN] Exception writing port image {save_path}: {e}")

            if total_ports_saved > 0:
                print(f"[OK] Saved {total_ports_saved} ports for unit {unit_id} to {os.path.join(unit_dir, 'ports')}")
        elif has_ports and not (has_switches or has_patches):
            # Skip saving ports for units that have no switch/patch panel
            print(f"[INFO] Skipping saving {len(unit_info.get('connected_port', [])) + len(unit_info.get('empty_port', []))} ports for unit {unit_id} (no device present)")
    
    print(f"[OK] Organized outputs saved to: {units_dir}")
    print(f"\n{'='*60}")
    print(f"[OK] Processing complete!")
    print(f"   Total units processed: {len(unit_data)}")
    print(f"   Switch groups: {len(switch_groups)}")
    print(f"   Patch panel groups: {len(patch_groups)}")

# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple rack images")
    parser.add_argument("--input", type=str, default=None, help="Input directory with images")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    # Override defaults if provided
    input_path = args.input or INPUT_DIR
    output_path = args.output or OUTPUT_ROOT
    
    print(f"[multii] INPUT_DIR: {input_path}")
    print(f"[multii] OUTPUT_ROOT: {output_path}")
    
    # Load models
    print(f"\n{'='*60}")
    print("Loading models...")
    models = load_models(MODEL_PATHS)
    
    if not models:
        print("[ERROR] No models loaded. Exiting.")
        sys.exit(1)
    
    # Process images
    print(f"{'='*60}")
    process_multi_images(input_path, output_path, models)
