import os
import base64
import torch
import pandas as pd
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from io import BytesIO

# =======================
# CONFIG (PIPELINE STYLE)
# =======================

DEFAULT_INPUT = "segmented_output/cables"
DEFAULT_OUTPUT = "Results"
HTML_NAME = "4_cable_classification.html"

device = "cpu"

# =======================
# HELPERS
# =======================

def normalize_key(x: str):
    return x.lower().strip().replace(" ", "_")

def image_to_base64(img_path, max_size=(128, 128)):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail(max_size)
    buf = BytesIO()
    img.save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def make_unique_key(final, ctype, usage, features, desc):
    return (
        (final or "").strip().lower(),
        (ctype or "").strip().lower(),
        (usage or "").strip().lower(),
        (features or "").strip().lower(),
        (desc or "").strip().lower(),
    )

# =======================
# TRANSFORMS
# =======================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def preprocess(p):
    return transform(Image.open(p).convert("RGB")).unsqueeze(0)

# =======================
# LOAD MODELS
# =======================

def load_model(path, arch):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # try to infer number of classes
    num_classes = None
    if isinstance(ckpt, dict) and "class_names" in ckpt:
        class_names = ckpt["class_names"]
        num_classes = len(class_names)
    else:
        # look for fc.weight shape in state dict
        for k, v in state.items():
            if k.endswith("fc.weight"):
                try:
                    num_classes = v.shape[0]
                except Exception:
                    pass
                break
        class_names = [str(i) for i in range(num_classes)] if num_classes is not None else None

    model = arch(weights=None)
    # ensure model.fc matches checkpoint
    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # attempt strict load first, then fallback to non-strict
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"[load_model] WARNING: strict load failed for {path}: {e}\nAttempting non-strict load (this may leave some layers randomly initialized)")
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e2:
            print(f"[load_model] ERROR: non-strict load also failed: {e2}")
            raise

    model.eval()
    return model, class_names

model1, classes1 = load_model(
    "Trained_Models/cable_prior.pth",
    models.resnet50
)

model2, classes2 = load_model(
    "Trained_Models/cable_medium.pth",
    models.resnet50
)

# =======================
# PREDICTION
# =======================

def predict(img_path):
    x = preprocess(img_path)
    with torch.no_grad():
        p1 = F.softmax(model1(x), dim=1)
        p2 = F.softmax(model2(x), dim=1)

    idx1 = p1.argmax().item()
    idx2 = p2.argmax().item()

    # same logic as your earlier scripts
    return classes1[idx1]

# =======================
# LOAD EXCEL
# =======================

df = pd.read_excel("Description_excel/Cable_Description.xlsx")
cable_info = {
    normalize_key(k): v
    for k, v in df.set_index("Cable_Name").to_dict("index").items()
}

# =======================
# MAIN
# =======================

def main():
    parser = argparse.ArgumentParser(description="Cable classification")
    parser.add_argument("--input", "-i", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    image_folder = args.input if args.input else DEFAULT_INPUT
    output_folder = args.output if args.output else DEFAULT_OUTPUT
    html_out = os.path.join(output_folder, HTML_NAME)

    print(f"[cable_match] image_folder={image_folder}")
    print(f"[cable_match] html_out={html_out}")

    if not os.path.exists(image_folder):
        os.makedirs(output_folder, exist_ok=True)
        with open(html_out, "w", encoding="utf-8") as f:
            f.write("<html><body><h2>No cable images found.</h2></body></html>")
        return

    os.makedirs(output_folder, exist_ok=True)

    rows = []
    seen_keys = set()

    for fname in os.listdir(image_folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_folder, fname)
        final = predict(img_path)
        img_b64 = image_to_base64(img_path)

        info = cable_info.get(normalize_key(final), {
            "Cable_Type": "N/A",
            "Usage": "N/A",
            "Key_Features": "N/A",
            "Description": "N/A"
        })

        unique_key = make_unique_key(
            final,
            info["Cable_Type"],
            info["Usage"],
            info["Key_Features"],
            info["Description"]
        )

        if unique_key in seen_keys:
            continue

        seen_keys.add(unique_key)

        rows.append((
            fname,
            img_b64,
            final,
            info["Cable_Type"],
            info["Usage"],
            info["Key_Features"],
            info["Description"]
        ))

    # =======================
    # HTML
    # =======================

    html = """
<html>
<head>
<style>
body { font-family: "Times New Roman", Times, serif; }
table { border-collapse: collapse; width: 100%; }
th, td {
    border: 1px solid black;
    padding: 8px;
    text-align: center;
}
th { background-color: #f2f2f2; }
img { max-width: 128px; max-height: 128px; }
.final {
    color: green;
    font-style: italic;
    font-weight: bold;
    font-size: 18px;
}
.info {
    max-width: 250px;
    word-wrap: break-word;
}
</style>
</head>
<body>
<h2>Cable Anslysis Report</h2>
<table>
<tr>
    <th>Cable</th>
    <th>Predicted Model</th>
    <th>Cable Type</th>
    <th>Usage</th>
    <th>Key Features</th>
    <th>Description</th>
</tr>
"""

    for fname, img_b64, final, ctype, usage, features, desc in rows:
        html += f"""
<tr>
    <td><img src="{img_b64}"></td>
    <td class="final">{final}</td>
    <td>{ctype}</td>
    <td>{usage}</td>
    <td>{features}</td>
    <td class="info">{desc}</td>
</tr>
"""

    html += "</table></body></html>"

    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html)

    print("OK: Unique cable classification report generated")
    print("Rows:", len(rows))
    print("Report:", html_out)

# =======================
# RUN
# =======================

if __name__ == "__main__":
    main()
