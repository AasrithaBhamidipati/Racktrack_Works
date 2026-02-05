import os
import re
import json
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import base64
import html as html_lib
from io import BytesIO
import torch.nn.functional as F
import pandas as pd
import difflib
import argparse

# ====================== DEDUP HELPERS ======================

def make_unique_key(final, switch_label, switch_name, port1, port2, parent_type):
    """
    Defines a logical unique port connection.
    """
    return (
        final.strip().lower(),
        (switch_label or "").strip().lower(),
        (switch_name or "").strip().lower(),
        (port1 or "").strip().lower(),
        (port2 or "").strip().lower(),
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

# ====================== TEXT HELPERS ======================

def normalize_key(s):
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r'\s+', ' ', s).strip().lower()

def find_port_info(norm_label, ports_lookup, cutoff=0.6):
    if norm_label in ports_lookup:
        return ports_lookup[norm_label], norm_label, False
    matches = difflib.get_close_matches(norm_label, ports_lookup.keys(), n=1, cutoff=cutoff)
    if matches:
        return ports_lookup[matches[0]], matches[0], True
    return {"Port 1":"N/A","Port 2":"N/A","Port Description":"N/A"}, None, False

# ====================== IMAGE UTILS ======================

def image_to_base64(img_path, max_size=(128,128)):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail(max_size)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ====================== MAIN ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="segmented_output/connected_port")
    parser.add_argument("--output", default="Results")
    parser.add_argument("--segmented-output", default="segmented_output")
    args = parser.parse_args()

    image_folder = args.input
    results_folder = args.output
    segmented_output = args.segmented_output

    html_out = os.path.join(results_folder, "5_connected_port_classification.html")
    os.makedirs(results_folder, exist_ok=True)

    # ====================== LOAD MODELS ======================

    checkpoint1 = torch.load("Trained_Models/cable_prior.pth", map_location="cpu")
    model1 = models.resnet50(weights=None)
    model1.fc = nn.Linear(model1.fc.in_features, len(checkpoint1["class_names"]))
    model1.load_state_dict(checkpoint1["model_state_dict"])
    model1.eval()

    checkpoint2 = torch.load("Trained_Models/cable_medium.pth", map_location="cpu")
    model2 = models.resnet50(weights=None)
    model2.fc = nn.Linear(model2.fc.in_features, checkpoint2["model_state_dict"]["fc.weight"].shape[0])
    model2.load_state_dict(checkpoint2["model_state_dict"])
    model2.eval()

    classes1 = checkpoint1["class_names"]
    classes2 = checkpoint2["class_names"]

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    def predict(img_path):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            p1 = F.softmax(model1(x), dim=1)
            p2 = F.softmax(model2(x), dim=1)
        return classes2[p2.argmax()] if p2.max()>p1.max() else classes1[p1.argmax()]

    # ====================== LOAD DATA ======================

    cable_info = pd.read_excel("Description_excel/Cable_Description.xlsx").set_index("Cable_Name").to_dict("index")
    ports_df = pd.read_excel("Description_excel/cables-ports.xlsx")
    ports_dict = {normalize_key(k):v for k,v in ports_df.set_index("Cable").to_dict("index").items()}

    # Load optional rack hierarchy produced by earlier steps; proceed if missing
    rack_hierarchy_path = os.path.join(segmented_output, "rack_hierarchy.json")
    if os.path.exists(rack_hierarchy_path):
        try:
            with open(rack_hierarchy_path, "r", encoding="utf-8") as f:
                rack_hierarchy = json.load(f)
        except Exception as e:
            print(f"[4_1_conneted_port_match] WARNING: failed to read {rack_hierarchy_path}: {e}")
            rack_hierarchy = {}
    else:
        print(f"[4_1_conneted_port_match] INFO: rack hierarchy file not found at {rack_hierarchy_path}; continuing with empty hierarchy.")
        rack_hierarchy = {}

    # ====================== PROCESS ======================

    rows = []
    seen_keys = set()

    for img in os.listdir(image_folder):
        if not img.lower().endswith((".jpg",".png",".jpeg")):
            continue

        img_path = os.path.join(image_folder, img)
        final = predict(img_path)

        info = cable_info.get(final, {})
        # Fallback descriptions for labels not present in the Excel
        FALLBACK_DESCRIPTIONS = {
            "hdmi a": "HDMI Type A connector — standard HDMI used for audio/video connections between devices. Supports digital video and audio signaling and is commonly used for monitors, TVs, and AV equipment.",
            "iec orange": "IEC power connector (orange) — standardized AC power connector (e.g., IEC C13/C14). The orange color is often used to indicate a specific circuit or PD/critical power line for servers and networking equipment.",
        }
        norm_final = normalize_key(final)
        port_info, matched, _ = find_port_info(norm_final, ports_dict)

        switch_label = rack_hierarchy.get(img, {}).get("parent_label", "")
        switch_name = rack_hierarchy.get(img, {}).get("parent_name", "")
        parent_type = rack_hierarchy.get(img, {}).get("parent_type", "")

        unique_key = make_unique_key(
            final, switch_label, switch_name,
            port_info.get("Port 1",""),
            port_info.get("Port 2",""),
            parent_type
        )

        if unique_key in seen_keys:
            continue

        seen_keys.add(unique_key)

        # Ensure description falls back to built-in text when missing from Excel
        desc_text = info.get("Description", "N/A")
        if (not desc_text or str(desc_text).strip().upper() == "N/A"):
            fb = FALLBACK_DESCRIPTIONS.get(normalize_key(final))
            if fb:
                desc_text = fb

        rows.append((
            img_path, final,
            info.get("Cable_Type","N/A"),
            desc_text,
            info.get("Usage","N/A"),
            port_info.get("Port 1","N/A"),
            port_info.get("Port Description","N/A"),
            port_info.get("Port 2","N/A"),
            switch_label,
            switch_name
        ))

    # ====================== HTML ======================

    html_content = """<html><body><h2>Connected Port Analysis Report</h2><table border=1>
    <tr><th>Image</th><th>Cable</th><th>Description</th><th>Usage</th><th>Port</th><th>Port Desc</th><th>Other Port</th></tr>
    """

    for r in rows:
        html_content += f"""
        <tr>
        <td><img src="{image_to_base64(r[0])}"></td>
        <td>{r[1]}</td>
        <td>{r[3]}</td>
        <td>{r[4]}</td>
        <td>{r[5]}</td>
        <td>{r[6]}</td>
        <td>{r[7]}</td>
        <td>{r[8]}</td>
        </tr>
        """

    html_content += "</table></body></html>"

    with open(html_out,"w",encoding="utf-8") as f:
        f.write(html_content)

    print("OK: Unique connected-port report generated")
    print("Rows:", len(rows))
    print("Report:", html_out)

if __name__ == "__main__":
    main()
