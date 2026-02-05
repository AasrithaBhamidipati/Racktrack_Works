import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import easyocr

# ---------------- CONFIG ----------------
ROOT_FOLDER = Path(r"S:\racktrack_video\IMG_0904")
MISPLACED_DIR = ROOT_FOLDER / "_misplaced_frames"
SIMILARITY_THRESHOLD = 0.75
OCR_TEXT_THRESHOLD = 8
IMG_SIZE = (224, 224)

MISPLACED_DIR.mkdir(exist_ok=True)

# ---------------- MODELS ----------------
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
ocr_reader = easyocr.Reader(['en'], gpu=False)

# ---------------- FUNCTIONS ----------------
def extract_embedding(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img, verbose=0)[0]

def is_ocr_frame(img_path):
    img = cv2.imread(str(img_path))
    texts = ocr_reader.readtext(img, detail=0)
    return sum(len(t) for t in texts) >= OCR_TEXT_THRESHOLD

# ---------------- STAGE 1: VISUAL SEPARATION ----------------
unit_embeddings = {}

for unit_folder in sorted(
    [f for f in ROOT_FOLDER.iterdir() if f.is_dir() and f.name != "_misplaced_frames"]
):
    print(f"\n[Stage 1] Processing {unit_folder.name}")

    image_paths = list(unit_folder.glob("*.jpg")) + list(unit_folder.glob("*.png"))
    if len(image_paths) < 2:
        continue

    embeddings = []
    for img in image_paths:
        emb = extract_embedding(img)
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    unit_center = np.mean(embeddings, axis=0)
    unit_embeddings[unit_folder] = unit_center

    for img_path, emb in zip(image_paths, embeddings):
        sim = cosine_similarity(unit_center.reshape(1, -1), emb.reshape(1, -1))[0][0]

        if sim < SIMILARITY_THRESHOLD:
            print(f"❌ Misplaced: {img_path.name}")
            shutil.move(str(img_path), MISPLACED_DIR / img_path.name)

# ---------------- STAGE 2: OCR RECOVERY ----------------
print("\n[Stage 2] OCR-based correction")

for img_path in MISPLACED_DIR.glob("*.*"):
    if not is_ocr_frame(img_path):
        continue

    print(f"📝 OCR detected: {img_path.name}")

    emb = extract_embedding(img_path)
    best_unit = None
    best_sim = 0

    for unit_folder, unit_center in unit_embeddings.items():
        sim = cosine_similarity(
            unit_center.reshape(1, -1),
            emb.reshape(1, -1)
        )[0][0]

        if sim > best_sim:
            best_sim = sim
            best_unit = unit_folder

    if best_unit:
        print(f"✅ Restored to {best_unit.name} (sim={best_sim:.2f})")
        shutil.move(str(img_path), best_unit / img_path.name)

print("\n✅ Two-stage frame correction completed")
