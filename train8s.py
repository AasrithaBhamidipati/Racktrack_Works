import os
import random
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Paths
# ---------------------------
DATA_YAML = r"D:\led\Led_Detection.v1i.yolov8\data.yaml" # change this if needed
MODEL_WEIGHTS = "yolov8s.pt"

# ---------------------------
# Device Selection
# ---------------------------
if torch.cuda.is_available():
    device = 0  # first CUDA device
    print("🚀 GPU detected. Training on GPU.")
else:
    device = "cpu"
    print("⚠️ GPU not found. Training on CPU.")

# ---------------------------
# Main execution
# ---------------------------
if __name__ == '__main__':
    # Load Model
    # ---------------------------
    model = YOLO(MODEL_WEIGHTS)

    # ---------------------------
    # Training Config
    # ---------------------------
    train_config = dict(
        data=DATA_YAML,
        epochs=50,
        batch=16,           # reduced from 32 for T1000 8GB GPU
        imgsz=512,          # reduced from 640 for GPU memory
        device=device,

        # Early stopping
        patience=5,

        # Performance
        workers=4,
        optimizer="auto",

        # Logging
        project="LED_Detection",
        name="yolov8s_led",
        exist_ok=True,

        # Augmentation
        degrees=10,
        fliplr=0.5,
        mosaic=1.0,

        # Mixed precision (only used when GPU is available)
        amp=torch.cuda.is_available()
    )

    # ---------------------------
    # Train
    # ---------------------------
    print("\nStarting Training...\n")

    for _ in tqdm(range(1), desc="Training Progress"):
        results = model.train(**train_config)

    print("\nTraining Complete ✅")
