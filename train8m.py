import random
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------
# Dataset Path
# ---------------------------
DATA_YAML = r"D:\led\Led_Detection.v1i.yolov8\data.yaml"# <-- CHANGE THIS

# Example:
# DATA_YAML = r"C:\Users\user\Led_Detect.v2i.yolov8\data.yaml"

# ---------------------------
# Device
# ---------------------------
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Main execution
# ---------------------------
if __name__ == '__main__':
    # Load Model (YOLOv8m better for LEDs)
    # ---------------------------
    model = YOLO("yolov8m.pt")

    # ---------------------------
    # Train
    # ---------------------------
    results = model.train(

        # dataset
        data=DATA_YAML,

        # training params
        epochs=15,
        batch=8,            # reduced further for T1000 8GB GPU
        imgsz=512,          # reduced from 800 for GPU memory

        # hardware
        device=device,
        workers=4,
        amp=False,          # disable AMP on T1000 to avoid OOM

        # early stopping
        patience=5,

        # augmentation (good for LEDs)
        mosaic=0.5,
        fliplr=0.5,
        degrees=5,

        # optimizer
        optimizer="auto",

        # logging
        project="LED_Detection",
        name="YOLOv8m_LED",
        exist_ok=True,

        # save
        save=True
    )

    print("Training complete ✅")
