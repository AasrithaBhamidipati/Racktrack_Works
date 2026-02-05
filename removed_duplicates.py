import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------- CONFIG ----------------
ROOT_FOLDER = Path(r"S:\racktrack_video\IMG_0902")
MISPLACED_DIR = ROOT_FOLDER / "_misplaced_frames"
SIMILARITY_THRESHOLD = 0.75   # adjust if needed
IMG_SIZE = (224, 224)

MISPLACED_DIR.mkdir(exist_ok=True)

# ---------------- MODEL ----------------
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# ---------------- FUNCTIONS ----------------
def extract_embedding(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img, verbose=0)[0]

# ---------------- PROCESS ----------------
for unit_folder in sorted([f for f in ROOT_FOLDER.iterdir() if f.is_dir() and f.name != "_misplaced_frames"]):
    print(f"\nProcessing {unit_folder.name}")

    image_paths = list(unit_folder.glob("*.jpg")) + list(unit_folder.glob("*.png"))
    if len(image_paths) < 2:
        continue

    embeddings = []
    for img in image_paths:
        emb = extract_embedding(img)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    # Unit reference embedding
    unit_center = np.mean(embeddings, axis=0).reshape(1, -1)

    for img_path, emb in zip(image_paths, embeddings):
        sim = cosine_similarity(unit_center, emb.reshape(1, -1))[0][0]

        if sim < SIMILARITY_THRESHOLD:
            print(f"❌ Misplaced: {img_path.name} (sim={sim:.2f})")
            shutil.move(str(img_path), MISPLACED_DIR / img_path.name)

print("\n✅ Frame separation completed")
