
import os
import cv2
import json
import shutil
import numpy as np
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
from pathlib import Path
 
 
 
# Only use command-line argument for input image
 
 
# --- Configuration ---
 
# --- Run if main ---
 
 
# --- Run if main ---
 
# Place this at the end of the file, after all function definitions
# Your trained models
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT.joinpath('Trained_Models')
MODEL_PATHS = {
    'cables': str(MODEL_DIR.joinpath('cable_best.pt')),
    'port': str(MODEL_DIR.joinpath('port_best.pt')),
    'rack': str(MODEL_DIR.joinpath('rack_best.pt')),
    'switch': str(MODEL_DIR.joinpath('switch_patch.pt')),
}
 
 
FINAL_CLASSES = ['cables', 'rack', 'patch_panel', 'switch', 'connected_port', 'empty_port']
CONF_THRESHOLDS = {"rack": 0.30, "switch": 0.30, "cables": 0.18}
MARGIN = 10
MIN_DIM = 128
 
# MODELS will be loaded on first run to give clearer errors when dependencies/models are missing
MODELS = None
 
 
def crop_and_save(region, path):
    """Crop and save with resize if too small"""
    h, w = region.shape[:2]
    if h < MIN_DIM or w < MIN_DIM:
        region = cv2.resize(region, (max(w, MIN_DIM), max(h, MIN_DIM)), interpolation=cv2.INTER_CUBIC)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, region, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
 
 
def boxes_overlap(port_box, parent_box):
    """
    Check if a port is properly contained within a parent device.
    A port must be fully or almost fully inside its parent device.
    Returns validation state as string:
    - "Valid": Port is properly contained
    - "Invalid Parent": No overlap with parent
    - "Port not contained": Poor containment or positioning
    """
    # Calculate port's area and intersection area
    port_area = (port_box[2] - port_box[0]) * (port_box[3] - port_box[1])
    
    # Calculate intersection
    xi1 = max(port_box[0], parent_box[0])
    yi1 = max(port_box[1], parent_box[1])
    xi2 = min(port_box[2], parent_box[2])
    yi2 = min(port_box[3], parent_box[3])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return "Invalid Parent"  # No overlap
        
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate what percentage of the port is inside the parent
    port_coverage = inter_area / port_area if port_area > 0 else 0
    
    # Port must be at least 85% inside the parent
    if port_coverage < 0.85:
        return "Port not contained"
        
    # Additional checks to prevent matching to components below:
    
    # 1. Check if port's center is inside parent
    port_center_x = (port_box[0] + port_box[2]) / 2
    port_center_y = (port_box[1] + port_box[3]) / 2
    if not (parent_box[0] <= port_center_x <= parent_box[2] and 
            parent_box[1] <= port_center_y <= parent_box[3]):
        return "Port not contained"
        
    # 2. Check vertical position - port should not be too close to bottom of parent
    parent_height = parent_box[3] - parent_box[1]
    rel_y = (port_box[3] - parent_box[1]) / parent_height
    if rel_y > 0.95:  # Port is too close to bottom edge
        return "Port not contained"
    
    return "Valid"  # All validation checks passed
 
 
def run_segmentation(image_path, output_dir=None):
    """Run segmentation on a single image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
 
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    if output_dir is None:
        output_dir = str(ROOT.joinpath('segmented_output'))
 
    # Save all outputs directly in class folders under output_dir
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    seg_out = output_dir
    for cls in FINAL_CLASSES:
        os.makedirs(os.path.join(seg_out, cls), exist_ok=True)
 
    # --- Structured JSON output ---
    structured_json = {
        "filename": image_name,
        "rack_bbox": None,
        "rack_segmented_filename": None,
        "switches": [],
        "patch_panels": [],
        "cables": [],
    }

    # Collect detections first (so ports detected earlier still get properly assigned)
    detections_by_class = {cls: [] for cls in FINAL_CLASSES}
    generic_counter = 0
    # --- Load models on first run ---
    global MODELS
    if YOLO is None:
        raise RuntimeError("Missing dependency: 'ultralytics'. Install with `pip install ultralytics` and required packages (opencv-python, numpy)")
    if MODELS is None:
        print("📦 Loading YOLO models...")
        MODELS = {}
        for cls, path in MODEL_PATHS.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}\nExpected in project Trained_Models folder. Place your .pt files there or update MODEL_PATHS.")
            MODELS[cls] = YOLO(path)

    # --- Run detection and save crops ---
    for target_class, model in MODELS.items():
        conf_threshold = CONF_THRESHOLDS.get(target_class, 0.2)
        results = model(img, conf=conf_threshold, verbose=False)[0]
        class_names = model.names
        print(f"[DEBUG] Results for model '{target_class}':")
        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            class_name_raw = class_names[int(cls_id)].lower()
            print(f"  Detected: class='{class_name_raw}', conf={conf:.3f}, box={list(map(int, box))}")
            # Flexible class name handling
            class_name = class_name_raw
            if class_name_raw in ["cable", "cables"]:
                class_name = "cables"
            if class_name_raw in ["rack", "racks"]:
                class_name = "rack"
            if class_name_raw in ["patch_panel", "patchpanel", "patch-panels", "patchpanels"]:
                class_name = "patch_panel"
            if class_name_raw in ["switch", "switches"]:
                class_name = "switch"
            if class_name_raw in ["connected_port", "connected_ports", "port_connected", "connected"]:
                class_name = "connected_port"
            if class_name_raw in ["empty_port", "empty_ports", "port_empty", "empty"]:
                class_name = "empty_port"

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
            x2, y2 = min(img.shape[1], x2 + MARGIN), min(img.shape[0], y2 + MARGIN)
            crop = img[y1:y2, x1:x2]

            seg_filename = f"{image_name}_{class_name}_{generic_counter}.jpg"
            out_path = os.path.join(seg_out, class_name, seg_filename)
            crop_and_save(crop, out_path)

            # Append to detections_by_class for post-processing
            detections_by_class.setdefault(class_name, []).append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "parent_image": image_name,
                "segmented_filename": seg_filename,
            })

            generic_counter += 1

    # --- Post-process detections into structured JSON with stable ordering ---
    object_counter = 0
    # First handle rack, switches, patch panels, cables so parents exist before ports
    for cls in ["rack", "switch", "patch_panel", "cables"]:
        for det in detections_by_class.get(cls, []):
            entry = {
                "id": f"{cls}_{object_counter}",
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "parent_image": det["parent_image"],
                "segmented_filename": det["segmented_filename"],
            }
            if cls == "rack":
                structured_json["rack_bbox"] = entry["bbox"]
                structured_json["rack_segmented_filename"] = entry["segmented_filename"]
            elif cls == "switch":
                entry["ports"] = {"connected": [], "empty": []}
                structured_json["switches"].append(entry)
            elif cls == "patch_panel":
                entry["ports"] = {"connected": [], "empty": []}
                structured_json["patch_panels"].append(entry)
            elif cls == "cables":
                structured_json["cables"].append(entry)
            object_counter += 1

            # Track excluded ports for logging
            excluded_ports = []
            
            # Now assign ports to their parent device (switch or patch_panel)
            for port_cls in ["connected_port", "empty_port"]:
                for det in detections_by_class.get(port_cls, []):
                    port_box = det["bbox"]
                    parent_found = False
                    validation_result = "Orphaned Port"  # Default state
                    
                    # First try switches
                    for switch in structured_json["switches"]:
                        validation = boxes_overlap(port_box, switch["bbox"])
                        if validation == "Valid":
                            port_entry = {
                                "bbox": det["bbox"],
                                "confidence": det["confidence"],
                                "parent_image": det["parent_image"],
                                "segmented_filename": det["segmented_filename"],
                                "parent_type": "switch",
                                "parent_id": switch.get("id"),
                            }
                            ptype = "connected" if port_cls == "connected_port" else "empty"
                            switch["ports"][ptype].append(port_entry)
                            if port_cls == "connected_port":
                                structured_json.setdefault("connected_ports", []).append(port_entry)
                            else:
                                structured_json.setdefault("empty_ports", []).append(port_entry)
                            parent_found = True
                            break
                        elif validation != "Invalid Parent":
                            validation_result = validation

                    # If not found in switches, try patch panels
                    if not parent_found:
                        for panel in structured_json["patch_panels"]:
                            validation = boxes_overlap(port_box, panel["bbox"])
                            if validation == "Valid":
                                port_entry = {
                                    "bbox": det["bbox"],
                                    "confidence": det["confidence"],
                                    "parent_image": det["parent_image"],
                                    "segmented_filename": det["segmented_filename"],
                                    "parent_type": "patch_panel",
                                    "parent_id": panel.get("id"),
                                }
                                ptype = "connected" if port_cls == "connected_port" else "empty"
                                panel["ports"][ptype].append(port_entry)
                                if port_cls == "connected_port":
                                    structured_json.setdefault("connected_ports", []).append(port_entry)
                                else:
                                    structured_json.setdefault("empty_ports", []).append(port_entry)
                                parent_found = True
                                break
                            elif validation != "Invalid Parent":
                                validation_result = validation

                    # If no valid parent found, keep the segmented image (do NOT delete).
                    # Add the port entry to structured_json as unassigned so downstream steps
                    # can still process or inspect these ports.
                    if not parent_found:
                        excluded_ports.append({
                            "filename": det["segmented_filename"],
                            "reason": validation_result
                        })
                        # Build an unassigned port entry (preserve metadata)
                        port_entry = {
                            "bbox": det["bbox"],
                            "confidence": det["confidence"],
                            "parent_image": det["parent_image"],
                            "segmented_filename": det["segmented_filename"],
                            "parent_type": None,
                            "parent_id": None,
                            "validation": validation_result
                        }
                        ptype = "connected" if port_cls == "connected_port" else "empty"
                        # Keep consistent with other parts of the JSON structure
                        if ptype == "connected":
                            structured_json.setdefault("connected_ports", []).append(port_entry)
                        else:
                            structured_json.setdefault("empty_ports", []).append(port_entry)
                            
            # Log excluded ports
            if excluded_ports:
                print("\n⚠️ The following ports were excluded due to invalid parent relationships:")
                for port in excluded_ports:
                    print(f"  • {port['filename']}: {port['reason']}")
                print()    # --- Save structured JSON ---
    with open(os.path.join(seg_out, f"{image_name}_structured.json"), 'w') as f:
        json.dump(structured_json, f, indent=2)
 
    print(f"✅ Segmentation complete for {image_path} → {seg_out}")
    return seg_out, structured_json
 
 
if __name__ == "__main__":
    import sys
    import glob
    import shutil
    if len(sys.argv) < 2:
        print("Usage: python single.py <image_folder> [output_dir]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(ROOT.joinpath('segmented_output'))
    # Clear only the class folders (not the entire output directory to preserve Results folder)
    for cls in FINAL_CLASSES:
        cls_path = os.path.join(output_dir, cls)
        if os.path.exists(cls_path):
            shutil.rmtree(cls_path)
    os.makedirs(output_dir, exist_ok=True)
    # If input_path is a directory, find the first image file
    if os.path.isdir(input_path):
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"):
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
        if not image_files:
            print(f"No image files found in {input_path}")
            sys.exit(1)
        image_path = image_files[0]
    else:
        image_path = input_path
    print(f"[INFO] Running segmentation on {image_path}, output to {output_dir}")
    try:
        # run_segmentation returns (seg_out, structured_json)
        seg_out, structured_json = run_segmentation(image_path, output_dir)
        print(f"[SUCCESS] Output saved to {seg_out}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(2)
 
 