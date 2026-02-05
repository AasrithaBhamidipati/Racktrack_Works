import os
import cv2
import json
from pathlib import Path
from datetime import datetime
import uuid
import shutil

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# CONFIG
# ============================================================

JOBS_OUTPUT_DIR = r"jobs_output"

MODEL_DIR = Path("Trained_Models")

MODELS_PATH = {
    "rack": MODEL_DIR / "rack_best.pt",
    "rack_unit": MODEL_DIR / "rack_unit.pt",
    "device": MODEL_DIR / "switch_patch.pt",
    "port": MODEL_DIR / "port_best.pt",
    "cable": MODEL_DIR / "cable_best.pt"
}

CONF = {
    "rack": 0.30,
    "rack_unit": 0.30,
    "device": 0.30,
    "port": 0.20,
    "cable": 0.18
}

RACK_UNITS = 42   # standard rack (fallback if no units detected)
MARGIN = 5


# ============================================================
# UTILS
# ============================================================

def crop(img, box):
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2]


def ensure(p):
    os.makedirs(p, exist_ok=True)


def create_job_output_dir():
    """Create a job output directory with timestamp and unique ID"""
    timestamp = int(datetime.now().timestamp() * 1000)  # milliseconds
    unique_id = str(uuid.uuid4())[:8]  # 8-char unique ID
    job_id = f"job_{timestamp}_{unique_id}"
    job_dir = os.path.join(JOBS_OUTPUT_DIR, job_id)
    ensure(job_dir)
    return job_dir, job_id


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(image_path, output_dir=None):

    if YOLO is None:
        raise RuntimeError("Install ultralytics first")

    img = cv2.imread(image_path)
    name = Path(image_path).stem

    # Create job output directory if not specified
    if output_dir is None:
        output_dir, job_id = create_job_output_dir()
    else:
        job_id = Path(output_dir).name

    out_root = output_dir

    # ---------------- LOAD MODELS ----------------
    models = {k: YOLO(str(v)) for k, v in MODELS_PATH.items()}

    # ---------------- DETECT RACK ----------------
    rack_res = models["rack"](img, conf=CONF["rack"], verbose=False)[0]
    rack_box = rack_res.boxes.xyxy[0]
    rack_img = crop(img, rack_box)

    rack_dir = os.path.join(out_root, "rack")
    ensure(rack_dir)
    cv2.imwrite(os.path.join(rack_dir, "rack.jpg"), rack_img)

    # Create units and cable folders
    units_dir = os.path.join(out_root, "units")
    # Clean up old units directory to remove stale data
    if os.path.exists(units_dir):
        import shutil
        shutil.rmtree(units_dir)
    ensure(units_dir)
    
    cables_dir = os.path.join(out_root, "cable")
    if os.path.exists(cables_dir):
        import shutil
        shutil.rmtree(cables_dir)
    ensure(cables_dir)

    # ---------------- DETECT RACK UNITS ----------------
    unit_res = models["rack_unit"](rack_img, conf=CONF["rack_unit"], verbose=False)[0]
    unit_boxes = unit_res.boxes.xyxy

    # Sort units by y-coordinate (top to bottom)
    unit_boxes = sorted(unit_boxes, key=lambda x: x[1])

    rack_json = {"units": []}

    # ---------------- SPLIT UNITS ----------------
    for u, unit_box in enumerate(unit_boxes):
        unit_img = crop(rack_img, unit_box)

        # ---------------- DEVICE ----------------
        dev_res = models["device"](unit_img, conf=CONF["device"], verbose=False)[0]

        if len(dev_res.boxes) > 0:
            cid = dev_res.boxes.cls[0]
            device_type_raw = dev_res.names[int(cid)]
            conf_score = dev_res.boxes.conf[0]
            
            print(f"[Unit {u+1}] Detected device: '{device_type_raw}' (confidence: {conf_score:.3f})")
            
            # Normalize device type for flexible matching
            device_type_lower = device_type_raw.lower()
            
            # Map various naming conventions to standard names (lowercase for storage)
            if device_type_lower in ["switch", "switches"]:
                device_type = "switch"
            elif device_type_lower in ["patch_panel", "patchpanel", "patch-panel", "patchpanels", "patch panel"]:
                device_type = "patchpanel"
            else:
                device_type = None  # Unrecognized device - skip this unit

            print(f"  → Normalized to: '{device_type}'")
            
            if device_type in ["switch", "patchpanel"]:
                print(f"  ✅ Processing {device_type} device")
                unit_id = f"unit{u+1}"

                unit_dir = os.path.join(units_dir, unit_id)
                ensure(unit_dir)

                # Remove old unit.jpg and unit.json if exist
                unit_jpg_path = os.path.join(unit_dir, "unit.jpg")
                unit_json_path = os.path.join(unit_dir, "unit.json")
                if os.path.exists(unit_jpg_path):
                    os.remove(unit_jpg_path)
                if os.path.exists(unit_json_path):
                    os.remove(unit_json_path)

                unit_json = {
                    "unit_id": u + 1,
                    "device": device_type,
                    "ports": {"connected": [], "empty": []},
                    "cables": []
                }

                box = dev_res.boxes.xyxy[0]
                dev_crop = crop(unit_img, box)

                # Remove old device folders (switch, patchpanel, patch_panel, etc.) if they exist
                old_dev_dirs = [
                    os.path.join(unit_dir, "device"),
                    os.path.join(unit_dir, "switch"),
                    os.path.join(unit_dir, "patchpanel"),
                    os.path.join(unit_dir, "patch_panel"),
                    os.path.join(unit_dir, "patchpanels"),
                    os.path.join(unit_dir, "switches"),
                ]
                for old_dir in old_dev_dirs:
                    if os.path.exists(old_dir):
                        import shutil
                        shutil.rmtree(old_dir)
                        print(f"    Cleaned up old folder: {old_dir}")

                dev_dir = os.path.join(unit_dir, device_type.lower().replace("_", ""))
                ensure(dev_dir)
                cv2.imwrite(os.path.join(dev_dir, f"{device_type.lower().replace('_', '')}.jpg"), dev_crop)

                # Normalize device type for flexible matching
                device_type_lower = device_type.lower()
                
                # ---------------- PORTS ----------------
                port_res = models["port"](dev_crop, conf=CONF["port"], verbose=False)[0]

                if len(port_res.boxes) > 0:
                    ports_dir = os.path.join(unit_dir, "ports")
                    ensure(ports_dir)
                    ensure(os.path.join(ports_dir, "connected"))
                    ensure(os.path.join(ports_dir, "empty"))

                    for i, (pbox, pcid) in enumerate(zip(port_res.boxes.xyxy, port_res.boxes.cls)):
                        pname = port_res.names[int(pcid)].lower()
                        
                        # Flexible port type detection
                        if "connected" in pname or "connected_port" in pname:
                            ptype = "connected"
                        elif "empty" in pname or "empty_port" in pname:
                            ptype = "empty"
                        else:
                            ptype = "empty"  # Default to empty if unclear

                        pcrop = crop(dev_crop, pbox)
                        fname = f"port_{ptype}_{i}.jpg"

                        cv2.imwrite(os.path.join(ports_dir, ptype, fname), pcrop)

                        unit_json["ports"][ptype].append({
                            "file": fname
                        })

                # ---------------- CABLES ----------------
                cable_res = models["cable"](unit_img, conf=CONF["cable"], verbose=False)[0]

                for i, cbox in enumerate(cable_res.boxes.xyxy):
                    ccrop = crop(unit_img, cbox)
                    fname = f"cable_unit{u+1}_{i}.jpg"
                    cv2.imwrite(os.path.join(cables_dir, fname), ccrop)

                    unit_json["cables"].append({
                        "file": fname
                    })

                # If a device was detected but no ports were found, do not store this unit
                has_ports = bool(unit_json["ports"]["connected"] or unit_json["ports"]["empty"])
                if not has_ports:
                    # Clean up created unit directory and any cable files written for this unit
                    print(f"[INFO] Skipping unit{u+1} - device detected but no ports present; removing stored files")
                    try:
                        if os.path.exists(unit_dir):
                            shutil.rmtree(unit_dir)
                    except Exception as e:
                        print(f"[WARN] Failed to remove unit dir {unit_dir}: {e}")

                    # Remove any cable files created for this unit
                    for c in unit_json.get("cables", []):
                        cf = os.path.join(cables_dir, c.get("file", ""))
                        try:
                            if os.path.exists(cf):
                                os.remove(cf)
                        except Exception as e:
                            print(f"[WARN] Failed to remove cable file {cf}: {e}")

                    # Do not append this unit to rack_json
                else:
                    rack_json["units"].append(unit_json)
            else:
                if device_type is None:
                    print(f"  ⚠️ Unrecognized device type: '{device_type_raw}' - skipping this unit")
                else:
                    print(f"  ⚠️ Invalid device type: '{device_type}' - skipping this unit")
        else:
            # No device, but still detect cables
            print(f"[Unit {u+1}] ⚠️ NO device detected in this unit")
            cable_res = models["cable"](unit_img, conf=CONF["cable"], verbose=False)[0]

            for i, cbox in enumerate(cable_res.boxes.xyxy):
                ccrop = crop(unit_img, cbox)
                fname = f"cable_unit{u+1}_{i}.jpg"
                cv2.imwrite(os.path.join(cables_dir, fname), ccrop)

    # ---------------- SAVE RACK JSON ----------------
    with open(os.path.join(rack_dir, "rack.json"), "w") as f:
        json.dump(rack_json, f, indent=2)

    print("✅ Unit-wise segmentation complete")
    print(f"📁 Job output: {output_dir}")
    print(f"🔑 Job ID: {job_id}")
    
    return output_dir, job_id


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unit-wise rack segmentation")
    parser.add_argument("--input", "-i", required=True, help="Input image path for rack segmentation")
    parser.add_argument("--output", "-o", default=None, help="Output directory for results (default: auto-generated job folder)")
    
    args = parser.parse_args()

    try:
        output_dir, job_id = run(args.input, args.output)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
