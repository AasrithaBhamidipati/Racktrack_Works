import os
import re
import json
import shutil
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torchvision import transforms, models
import torch.nn.functional as F
import base64
from io import BytesIO
import argparse

# image extensions used when searching for parent thumbnails
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# ====================== DEDUP HELPERS ======================

def make_unique_key(predicted_class, parent_label, parent_name, parent_type):
    return (
        (predicted_class or "").strip().lower(),
        (parent_label or "").strip().lower(),
        (parent_name or "").strip().lower(),
        (parent_type or "").strip().lower()
    )

# ====================== IOU VALIDATION ======================

def boxes_overlap(port_box, parent_box):
    port_area = (port_box[2] - port_box[0]) * (port_box[3] - port_box[1])
    xi1 = max(port_box[0], parent_box[0])
    yi1 = max(port_box[1], parent_box[1])
    xi2 = min(port_box[2], parent_box[2])
    yi2 = min(port_box[3], parent_box[3])

    if xi2 <= xi1 or yi2 <= yi1:
        return "Invalid Parent"

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    if inter_area / port_area < 0.85:
        return "Port not contained within any device"

    cx = (port_box[0] + port_box[2]) / 2
    cy = (port_box[1] + port_box[3]) / 2
    if not (parent_box[0] <= cx <= parent_box[2] and parent_box[1] <= cy <= parent_box[3]):
        return "Port not contained within any device"

    if (port_box[3] - parent_box[1]) / (parent_box[3] - parent_box[1]) > 0.95:
        return "Port not contained within any device"

    return "Valid"

# ====================== IMAGE UTILS ======================

def image_to_base64(img_path, bbox=None, max_size=(128, 128), draw_bbox=False):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail(max_size)

    if bbox is not None and draw_bbox:
        orig_size = Image.open(img_path).size
        scale_x = img.size[0] / orig_size[0]
        scale_y = img.size[1] / orig_size[1]
        scaled_bbox = [
            int(bbox[0] * scale_x),
            int(bbox[1] * scale_y),
            int(bbox[2] * scale_x),
            int(bbox[3] * scale_y),
        ]
        draw = ImageDraw.Draw(img)
        draw.rectangle(scaled_bbox, outline="red", width=2)

    buf = BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ====================== MAIN ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="segmented_output/empty_port")
    parser.add_argument("--output", default="Results")
    parser.add_argument("--segmented-output", default="segmented_output")
    parser.add_argument('--verbose', action='store_true', help='Verbose diagnostics')
    parser.add_argument("--job", "-J", default=None, help="Path to job folder (script will use its output/empty_port and output/Results)")
    parser.add_argument("--structured-json", "-j", default=None,
                        help="Path to a specific structured JSON to use for parent lookup")
    parser.add_argument("--structured-dir", "-d", default=None,
                        help="Directory to search for structured JSON files (overrides searching parent folders)")
    args = parser.parse_args()
    VERBOSE = getattr(args, 'verbose', False)

    image_folder = args.input
    results_folder = args.output
    segmented_output = args.segmented_output

    # If a job root is given and defaults were left, set defaults based on job folder
    if args.job:
        job_root = args.job
        # prefer absolute paths
        job_root = os.path.abspath(job_root)
        candidate_input = os.path.join(job_root, 'output', 'empty_port')
        candidate_results = os.path.join(job_root, 'output', 'Results')
        if (image_folder == 'segmented_output/empty_port' or not os.path.exists(image_folder)) and os.path.exists(candidate_input):
            image_folder = candidate_input
        if (results_folder == 'Results' or not os.path.exists(results_folder)):
            results_folder = candidate_results
        # if structured-json not provided explicitly, try to find one under job/output
        if not args.structured_json:
            out_dir = os.path.join(job_root, 'output')
            if os.path.exists(out_dir):
                for fname in os.listdir(out_dir):
                    if fname.lower().endswith('structured.json'):
                        args.structured_json = os.path.join(out_dir, fname)
                        print(f"[4_port_match] Auto-discovered structured JSON: {args.structured_json}")
                        break

    # If user supplied a specific structured JSON, try to load it upfront
    structured_json_content = None
    if args.structured_json:
        try:
            if os.path.exists(args.structured_json):
                with open(args.structured_json, 'r', encoding='utf-8') as jf:
                    structured_json_content = json.load(jf)
                print(f"[4_port_match] Loaded structured JSON from: {args.structured_json}")
            else:
                print(f"[4_port_match] Warning: structured JSON not found: {args.structured_json}")
        except Exception as e:
            print(f"[4_port_match] Warning: failed to load structured JSON {args.structured_json}: {e}")

    # validate input early
    if not os.path.exists(image_folder):
        print(f"[4_port_match] ERROR: input folder not found: {image_folder}")
        return

    html_path = os.path.join(results_folder, "5_Ports_Results.html")
    os.makedirs(results_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================== MODEL ======================

    class_names = [
        'BNC_Female','CX4','DC_BARREL','DISPLAYPORT','DVI','Ethernet','FC_APC_UPC',
        'F_Type','FireWire','GPON_OLT','GPON_SC_APC','HDMI_A','IEC_C13_C14',
        'IEC_C15_C16','IEC_C19_C20','IEC_C5_C6','IEC_C7_C8','LC_APC','LC_UPC',
        'LSH_APC','MICRO_HDMI_D','MINI_DISPLAY','MINI_HDMI_C','MTP_APC','MTP_UPC',
        'NEMA_5_15','NEMA_l6_30R','OSFP','PPS','PS_2','QSFP','QSFP_DD','RF_Coax',
        'RS_232','SC_APC','SC_UPC','SFP','SFP_DD','ST','Stacking_ports',
        'USB_2_B','USB_2_B_MICRO','USB_3_A','USB_3_B','USB_3_B_MICRO','USB_A',
        'USB_C','USB_MINI_B','VGA','XFP'
    ]

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("Trained_Models/port_identify.pth", map_location="cpu"))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    def predict(img_path):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).cpu().numpy().flatten()
        idx = probs.argmax()
        return class_names[idx], probs[idx]

    # ====================== LOAD DATA ======================
    # Try several candidate filenames for description and cable mappings
    def _read_first_excel(candidates):
        for c in candidates:
            if os.path.exists(c):
                try:
                    df = pd.read_excel(c)
                    print(f"[4_port_match] Loaded Excel: {c}")
                    return df
                except Exception as e:
                    print(f"[4_port_match] WARNING: failed to read {c}: {e}")
        print(f"[4_port_match] WARNING: no description Excel found among: {candidates}")
        return pd.DataFrame()

    desc_candidates = [
        "Description_excel/Port_Descriptions.xlsx",
        "Description_excel/Port-Descriptions.xlsx",
        "Description_excel/port_descriptions.xlsx",
        "Description_excel/PortDescriptions.xlsx",
    ]
    desc_df = _read_first_excel(desc_candidates)
    if not desc_df.empty and "Port" in desc_df.columns and "Description" in desc_df.columns:
        description_map = dict(zip(desc_df["Port"], desc_df["Description"]))
    else:
        description_map = {}

    cable_candidates = [
        "Description_excel/Port-cable.xlsx",
        "Description_excel/port-cable.xlsx",
        "Description_excel/cables-ports.xlsx",
        "Description_excel/ports-cable.xlsx",
        "Description_excel/Port_Cable.xlsx",
    ]
    cable_df = _read_first_excel(cable_candidates)
    if not cable_df.empty and cable_df.shape[1] >= 2:
        port_cable_map = dict(zip(cable_df.iloc[:,0].astype(str), cable_df.iloc[:,1].astype(str)))
    else:
        port_cable_map = {}

    # Normalize mapping keys so lookups match model class names that may
    # differ by punctuation/whitespace/case (e.g. USB_A vs USB A / USB-A).
    def normalize_key(s):
        if s is None:
            return ""
        return re.sub(r'[^a-z0-9]', '', str(s).lower())

    normalized_port_cable_map = {normalize_key(k): v for k, v in port_cable_map.items()}
    # also normalize description keys for consistent lookup
    normalized_description_map = {normalize_key(k): v for k, v in description_map.items()}

    def get_expected_cable_for(pred_class):
        nk = normalize_key(pred_class)
        # exact match
        if nk in normalized_port_cable_map:
            return normalized_port_cable_map[nk]
        # substring/partial match: try to find a mapping key that contains or is contained by nk
        for k, v in normalized_port_cable_map.items():
            if k and (k in nk or nk in k):
                return v
        # Alias-based fallback: common groups -> prefer a mapping key prefix
        alias_prefixes = {
            'ethernet': ['rj45'],
            'usb': ['usb'],
            'sc': ['sc'],
            'lc': ['lc'],
            'hdmi': ['hdmia', 'hdmi'],
            'bnc': ['bnc'],
            'ps2': ['ps2'],
            'nema': ['nema'],
            'rs232': ['rs232'],
            'sfp': ['sfp'],
            'xfp': ['xfp']
        }
        for alias, prefixes in alias_prefixes.items():
            if alias in nk:
                for pref in prefixes:
                    for k, v in normalized_port_cable_map.items():
                        if k.startswith(pref):
                            return v
        # as a last resort, return empty string
        return ""

    def normalize_cable_label(raw):
        if not raw:
            return ""
        s = str(raw).strip()
        nk = normalize_key(s)
        # common mappings
        if nk.startswith('rj45'):
            return 'RJ45'
        if nk.startswith('hdmia') or nk.startswith('hdmi'):
            return 'HDMI'
        if nk.startswith('lc'):
            return 'LC'
        if nk.startswith('sc'):
            return 'SC'
        if nk.startswith('bnc'):
            return 'BNC'
        if nk.startswith('usb'):
            return 'USB'
        if nk.startswith('ps2'):
            return 'PS/2'
        if nk.startswith('nema'):
            return 'NEMA'
        if nk.startswith('rs232') or 'rs232' in nk:
            return 'RS-232'
        if nk.startswith('sfp'):
            return 'SFP'
        if nk.startswith('xfp'):
            return 'XFP'
        # fallback: title-case the raw string
        return s.title()

    # ====================== PROCESS ======================

    results = []
    seen_keys = set()
    missing_expected = {}

    for fname in os.listdir(image_folder):
        if not fname.lower().endswith((".jpg",".png",".jpeg")):
            continue

        # extract an image id prefix (used by structured json filenames)
        img_id = os.path.splitext(fname)[0]
        image_path = os.path.join(image_folder, fname)
        pred_class, conf = predict(image_path)

        # ---- parent lookup ----
        parent_label, parent_name, parent_type = "Unknown", "", ""
        parent_node = None
        try:
            data = None
            # If a specific structured JSON was provided, use it
            if structured_json_content:
                data = structured_json_content
            else:
                # Build candidate directories to search
                parent_dir = os.path.dirname(image_folder)
                search_dirs = []
                if args.structured_dir:
                    search_dirs.append(args.structured_dir)
                search_dirs.append(parent_dir)
                search_dirs.append(os.path.abspath(os.path.join(parent_dir, '..')))

                for sd in search_dirs:
                    if not sd or not os.path.exists(sd):
                        continue
                    for candidate in os.listdir(sd):
                        lc = candidate.lower()
                        if lc.endswith("structured.json") and (candidate.startswith(img_id) or img_id in candidate):
                            json_path = os.path.join(sd, candidate)
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                print(f"[4_port_match] Found structured JSON: {json_path}")
                                break
                            except Exception:
                                data = None
                    if data:
                        break

            parent_seg_name = None
            # keywords indicating device-level segmented images
            DEVICE_KEYWORDS = ('switch', 'patch_panel', 'patchpanel', 'panel', 'rack', 'chassis')

            def is_device_node(node):
                if not node or not isinstance(node, dict):
                    return False
                # direct indicator: has 'ports' collection
                if node.get('ports'):
                    return True
                # type/category/parent_type fields
                for fld in ('type', 'category', 'parent_type'):
                    v = node.get(fld)
                    if isinstance(v, str) and any(k in v.lower() for k in DEVICE_KEYWORDS):
                        return True
                # id may contain device keywords
                nid = node.get('id', '')
                if isinstance(nid, str) and any(k in nid.lower() for k in DEVICE_KEYWORDS):
                    return True
                return False

            def find_first_device(node):
                if isinstance(node, dict):
                    if is_device_node(node):
                        return node
                    for v in node.values():
                        res = find_first_device(v)
                        if res:
                            return res
                elif isinstance(node, list):
                    for item in node:
                        res = find_first_device(item)
                        if res:
                            return res
                return None

            def find_device_for_port(root, port_seg_name):
                # traverse and find a node that contains this port in its 'ports' structure
                if not root:
                    return None
                if isinstance(root, dict):
                    # if this node has ports dict/list, search inside
                    ports = root.get('ports')
                    if ports:
                        # ports may be dict with keys like 'empty','connected'
                        if isinstance(ports, dict):
                            for v in ports.values():
                                if isinstance(v, list):
                                    for item in v:
                                        seg = item.get('segmented_filename') or item.get('segmented_image')
                                        if seg and os.path.basename(seg) == port_seg_name:
                                            return root
                        elif isinstance(ports, list):
                            for item in ports:
                                seg = item.get('segmented_filename') or item.get('segmented_image')
                                if seg and os.path.basename(seg) == port_seg_name:
                                    return root
                    # otherwise recurse
                    for v in root.values():
                        res = find_device_for_port(v, port_seg_name)
                        if res:
                            return res
                elif isinstance(root, list):
                    for item in root:
                        res = find_device_for_port(item, port_seg_name)
                        if res:
                            return res
                return None
            def find_node_by_id(node, id_val):
                if isinstance(node, dict):
                    if node.get('id') == id_val:
                        return node
                    for v in node.values():
                        res = find_node_by_id(v, id_val)
                        if res:
                            return res
                elif isinstance(node, list):
                    for item in node:
                        res = find_node_by_id(item, id_val)
                        if res:
                            return res
                return None

            if data:
                # traverse the JSON and find any dicts that include a segmented_filename
                def walk_for_segmented(node):
                    if isinstance(node, dict):
                        if 'segmented_filename' in node or 'segmented_image' in node:
                            yield node
                        for v in node.values():
                            yield from walk_for_segmented(v)
                    elif isinstance(node, list):
                        for item in node:
                            yield from walk_for_segmented(item)

                for p in walk_for_segmented(data):
                    seg = p.get('segmented_filename') or p.get('segmented_image') or ''
                    seg_base = os.path.basename(seg) if seg else ''
                    if not seg:
                        continue
                    # normalize and match exact or substring
                    # broaden matching: compare basenames and allow partial matches both ways
                    if seg == fname or seg_base == fname or seg.endswith(fname) or fname.endswith(seg) or fname in seg or seg in fname or seg_base in fname or fname in seg_base:
                        parent_type = p.get('parent_type')
                        parent_label = parent_type.upper() if parent_type else 'Unknown'
                        parent_name = p.get('parent_id', '')
                        # try to find the parent node in the JSON to get its segmented image
                        if parent_name:
                            parent_node = find_node_by_id(data, parent_name)
                            if parent_node:
                                parent_seg_name = parent_node.get('segmented_filename') or parent_node.get('segmented_image')
                                if parent_seg_name:
                                    parent_seg_name = os.path.basename(parent_seg_name)
                                # if parent's segmented filename looks like a port image (no device keyword),
                                # try to find the device node that contains this port
                                if parent_seg_name and not any(k in parent_seg_name.lower() for k in DEVICE_KEYWORDS):
                                        # try to find device ancestor for this port; prefer device segmented image
                                        dev = find_device_for_port(data, os.path.basename(seg if seg else fname))
                                        if dev and is_device_node(dev):
                                            parent_node = dev
                                            parent_seg_name = (dev.get('segmented_filename') or dev.get('segmented_image'))
                                            if parent_seg_name:
                                                parent_seg_name = os.path.basename(parent_seg_name)
                                        else:
                                            # keep original parent_node (may be non-device) for now
                                            parent_node = parent_node
                                else:
                                    # parent_seg_name already device-like
                                    parent_node = parent_node
                        # If parent_id missing or parent is not a device, try to find a device ancestor
                        if (not parent_name) or (parent_node and not is_device_node(parent_node)):
                            # try to find device that contains this port
                            dev = find_device_for_port(data, os.path.basename(seg if seg else fname))
                            if dev and is_device_node(dev):
                                parent_node = dev
                                parent_name = dev.get('id','')
                                parent_type = dev.get('parent_type') or dev.get('type') or dev.get('category') or ''
                                parent_label = parent_type.upper() if parent_type else 'Unknown'
                                parent_seg_name = (dev.get('segmented_filename') or dev.get('segmented_image'))
                                if parent_seg_name:
                                    parent_seg_name = os.path.basename(parent_seg_name)
                        # If still no device parent, pick first device node in JSON as a last resort
                        if (not parent_node) or (parent_node and not is_device_node(parent_node)):
                            first_dev = find_first_device(data)
                            if first_dev and is_device_node(first_dev):
                                parent_node = first_dev
                                parent_name = parent_name or first_dev.get('id','')
                                parent_type = parent_type or first_dev.get('parent_type') or first_dev.get('type') or first_dev.get('category') or ''
                                parent_label = parent_label or (parent_type.upper() if parent_type else 'Unknown')
                                parent_seg_name = parent_seg_name or (first_dev.get('segmented_filename') or first_dev.get('segmented_image'))
                                if parent_seg_name:
                                    parent_seg_name = os.path.basename(parent_seg_name)

                        # If parent_id missing, try bbox/IoU fallback to find a containing parent
                        if (not parent_name) and p.get('bbox'):
                            port_bbox = p.get('bbox')
                            # collect candidate parent nodes with bbox and id
                            def collect_parents(node, acc):
                                if isinstance(node, dict):
                                    if node.get('id') and node.get('bbox'):
                                        acc.append(node)
                                    for v in node.values():
                                        collect_parents(v, acc)
                                elif isinstance(node, list):
                                    for item in node:
                                        collect_parents(item, acc)
                            parents = []
                            collect_parents(data, parents)
                            # try to find a parent whose bbox contains the port (using boxes_overlap)
                            for cand in parents:
                                try:
                                    cand_bbox = cand.get('bbox')
                                    if not cand_bbox:
                                        continue
                                    res = boxes_overlap(port_bbox, cand_bbox)
                                    if res == 'Valid':
                                        # ensure candidate is a device (has ports) otherwise try to find its device ancestor
                                        if is_device_node(cand):
                                            chosen = cand
                                        else:
                                            # try to find device ancestor for this candidate
                                            dev = find_device_for_port(data, os.path.basename(seg if seg else fname))
                                            chosen = dev if (dev and is_device_node(dev)) else cand
                                        parent_name = chosen.get('id', '')
                                        parent_type = chosen.get('parent_type') or chosen.get('type') or chosen.get('category') or ''
                                        parent_label = parent_type.upper() if parent_type else 'Unknown'
                                        parent_seg_name = chosen.get('segmented_filename') or chosen.get('segmented_image')
                                        if parent_seg_name:
                                            parent_seg_name = os.path.basename(parent_seg_name)
                                        break
                                except Exception:
                                    continue
                        break
        except Exception:
            pass

        # Try to find a parent thumbnail image: search job input, segmented_output, and parent folders.
        parent_thumb_b64 = None
        try:
            search_roots = [os.path.dirname(image_folder), segmented_output, os.path.join(os.path.dirname(image_folder), '..')]
            found = None
            for root in search_roots:
                if not root:
                    continue
                root = os.path.abspath(root)
                if not os.path.exists(root):
                    continue
                for dirpath, dirnames, filenames in os.walk(root):
                    for fn in filenames:
                        # skip the port image itself
                        if fn == fname:
                            continue
                        if not fn.lower().endswith(IMG_EXTS):
                            continue
                        # if we found the parent's segmented filename from JSON, prefer exact or substring matches
                        if parent_seg_name:
                            if fn == parent_seg_name or fn.endswith(parent_seg_name) or parent_seg_name in fn:
                                # only accept parent_seg_name matches if the parent is device-like
                                if any(k in parent_seg_name.lower() for k in DEVICE_KEYWORDS) or (parent_node and is_device_node(parent_node)):
                                    found = os.path.join(dirpath, fn)
                                    break

                        lower = fn.lower()
                        # match by parent_name/label only when those refer to device nodes
                        if parent_name and parent_node and is_device_node(parent_node) and parent_name.lower() in lower:
                            found = os.path.join(dirpath, fn)
                            break
                        if parent_label and any(k in parent_label.lower() for k in DEVICE_KEYWORDS) and parent_label.lower() in lower:
                            found = os.path.join(dirpath, fn)
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                try:
                    parent_thumb_b64 = image_to_base64(found, max_size=(120,90))
                except Exception:
                    parent_thumb_b64 = None
        except Exception:
            parent_thumb_b64 = None

        if VERBOSE:
            print(f"[4_port_match] File={fname} parent_name={parent_name} parent_label={parent_label} parent_seg_name={parent_seg_name} parent_node_found={'yes' if parent_node else 'no'} thumbnail_found={'yes' if parent_thumb_b64 else 'no'}")

        # Additional fallback: if we have structured JSON, try to find any device-level segmented
        # image mentioned in the JSON (nodes that look like devices) and match those filenames
        # to files under the search roots. This helps when parent_seg_name is missing or not
        # present on disk but a device-level segmented image exists elsewhere.
        if not parent_thumb_b64 and 'data' in locals() and data:
            try:
                device_seg_candidates = []
                def collect_device_segmented(node):
                    if isinstance(node, dict):
                        if is_device_node(node):
                            seg = node.get('segmented_filename') or node.get('segmented_image')
                            if seg:
                                device_seg_candidates.append(os.path.basename(seg))
                        for v in node.values():
                            collect_device_segmented(v)
                    elif isinstance(node, list):
                        for item in node:
                            collect_device_segmented(item)

                collect_device_segmented(data)

                if device_seg_candidates:
                    # search for any of these filenames in the search roots
                    for root in search_roots:
                        if not root:
                            continue
                        root = os.path.abspath(root)
                        if not os.path.exists(root):
                            continue
                        for dirpath, dirnames, filenames in os.walk(root):
                            for fn in filenames:
                                if not fn.lower().endswith(IMG_EXTS):
                                    continue
                                for cand in device_seg_candidates:
                                    if not cand:
                                        continue
                                    if fn == cand or fn.endswith(cand) or cand in fn or os.path.basename(fn) == cand:
                                        try:
                                            parent_thumb_b64 = image_to_base64(os.path.join(dirpath, fn), max_size=(120,90))
                                            raise StopIteration
                                        except StopIteration:
                                            break
                            if parent_thumb_b64:
                                break
                        if parent_thumb_b64:
                            break
            except Exception:
                parent_thumb_b64 = parent_thumb_b64

        if VERBOSE and not parent_thumb_b64:
            print(f"[4_port_match] Fallback device candidates count={len(device_seg_candidates) if 'device_seg_candidates' in locals() else 0}")

        # Fallback: if no thumbnail found but we have structured JSON and parent_node with bbox,
        # try cropping the rack segmented image (data['rack_segmented_filename']) or parent's segmented image
        if not parent_thumb_b64 and 'data' in locals() and data and parent_node:
            try:
                bbox = parent_node.get('bbox')
                if bbox:
                    # try candidates: parent's segmented filename, top-level rack_segmented_filename
                    rack_candidates = []
                    pn_seg = parent_node.get('segmented_filename') or parent_node.get('segmented_image')
                    if pn_seg:
                        rack_candidates.append(os.path.basename(pn_seg))
                    top_rack = data.get('rack_segmented_filename') or data.get('rack_segmented_image')
                    if top_rack:
                        rack_candidates.append(os.path.basename(top_rack))

                    found_img = None
                    for root in [os.path.dirname(image_folder), segmented_output, os.path.join(os.path.dirname(image_folder), '..')]:
                        if not root:
                            continue
                        root = os.path.abspath(root)
                        for rc in rack_candidates:
                            if not rc:
                                continue
                            candidate_path = os.path.join(root, rc)
                            if os.path.exists(candidate_path):
                                found_img = candidate_path
                                break
                        if found_img:
                            break

                    if found_img:
                        img = Image.open(found_img).convert('RGB')
                        # ensure bbox values are ints and within image bounds
                        x1,y1,x2,y2 = map(int, bbox)
                        x1 = max(0, min(x1, img.width-1))
                        x2 = max(0, min(x2, img.width))
                        y1 = max(0, min(y1, img.height-1))
                        y2 = max(0, min(y2, img.height))
                        if x2>x1 and y2>y1:
                            crop = img.crop((x1,y1,x2,y2))
                            buf = BytesIO()
                            crop.thumbnail((120,90))
                            crop.save(buf, format='JPEG')
                            parent_thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
            except Exception:
                parent_thumb_b64 = parent_thumb_b64

        # -------- DEDUP --------
        unique_key = make_unique_key(pred_class, parent_label, parent_name, parent_type)
        if unique_key in seen_keys:
            continue
        seen_keys.add(unique_key)

        expected_cable_val = get_expected_cable_for(pred_class)
        if expected_cable_val:
            expected_cable_label = normalize_cable_label(expected_cable_val)
        else:
            # derive a sensible cable label from the predicted class when mapping missing
            expected_cable_label = normalize_cable_label(pred_class)
            # store something useful in the raw field as well
            expected_cable_val = expected_cable_label

        if not expected_cable_val:
            # record for diagnostics: original class -> normalized key
            missing_expected[pred_class] = normalize_key(pred_class)

        results.append({
            "fname": fname,
            "port_thumbnail": image_to_base64(image_path, max_size=(64,64)),
            "predicted_class": pred_class,
            "expected_cable": expected_cable_val,
            "expected_cable_label": expected_cable_label,
            "description": normalized_description_map.get(normalize_key(pred_class), description_map.get(pred_class, "")),
            "parent_label": parent_label,
            "parent_name": parent_name,
            "parent_type": parent_type,
            "parent_thumbnail": parent_thumb_b64
        })

    # ====================== HTML ======================

    html = """<html><body><h2>Empty Port Analysis Report</h2>
    <table border=1 width=95% align=center>
    <tr>
        <th>Port</th><th>Predicted Port</th><th>Expected Cable</th>
        <th>Description</th>
    </tr>
    """

    for r in results:
        parent_cell = ''
        # show only the parent image; if not available, show port area as fallback image
        if r.get('parent_thumbnail'):
            parent_cell = f"<img src=\"{r['parent_thumbnail']}\" width=120>"
        else:
            # Do not show port images as parent device; leave blank when no device image found
            parent_cell = ""

        html += f"""
        <tr>
            <td><img src="{r['port_thumbnail']}" width=64></td>
            <td>{r['predicted_class']}</td>
            <td>{r['expected_cable_label']}</td>
            <td>{r['description']}</td>
            
        </tr>
        """

    html += "</table></body></html>"

    with open(html_path,"w",encoding="utf-8") as f:
        f.write(html)

    print("OK: Unique empty-port report generated")
    print(f"Rows: {len(results)}")
    print(f"Report: {html_path}")

    # Diagnostic output: which predicted classes had no expected cable mapping
    if missing_expected:
        print("[4_port_match] Diagnostic: predicted classes with no expected cable mapping:")
        for pc, nk in missing_expected.items():
            present = 'YES' if nk in normalized_port_cable_map else 'NO'
            print(f"  {pc} -> normalized='{nk}' in_mappings={present}")
        print(f"[4_port_match] Available normalized mapping keys (sample): {list(normalized_port_cable_map.keys())[:30]}")
    # Print mapping of segmented file -> parent device (from JSON) for quick debugging
    print('[4_port_match] File -> parent_id mappings:')
    for r in results:
        print(f"  {r['fname']} -> {r.get('parent_name','Unknown')} ({r.get('parent_label','')})")

    

if __name__ == "__main__":
    main()
