import os
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm

def calculate_sharpness(image_path):
    """Returns sharpness score using Laplacian variance"""
    image = cv2.imread(image_path)
    if image is None:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_best_frame(frames_dir, best_frame_output_dir=None, full_frames_output_dir=None):
    """Find the sharpest frame from extracted frames, move full rack images to full_frames.

    If `best_frame_output_dir` or `full_frames_output_dir` are not provided, this
    function will create sibling folders `04_best_frame` and `03_full_frames`
    next to the parent video folder of `frames_dir`.
    """
    frames_dir = Path(frames_dir)
    
    if not frames_dir.exists():
        print(f"[✗] Frames directory not found: {frames_dir}")
        return

    # Get all frame files (png or jpg)
    frame_files = sorted([f for f in frames_dir.glob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    
    if not frame_files:
        print(f"[✗] No frames found in: {frames_dir}")
        return

    print(f"[✔] Found {len(frame_files)} frames\n")

    best_frame = None
    best_score = 0
    best_filename = None
    full_rack_frames = []
    unit_frames = []

    # Calculate sharpness for each frame
    for frame_file in tqdm(frame_files, desc="Calculating frame sharpness"):
        score = calculate_sharpness(str(frame_file))
        print(f"{frame_file.name:30} → Sharpness: {score:.2f}")

        # Classify frames: full rack (sharpness > 50) vs unit-wise (sharpness <= 50)
        if score > 50:
            full_rack_frames.append((frame_file, score))
        else:
            unit_frames.append((frame_file, score))

        if score > best_score:
            best_score = score
            best_frame = frame_file
            best_filename = frame_file.name

    if best_frame is None:
        print("[✗] Could not analyze any frames")
        return

    # Determine video root (parent folder of frames folder)
    video_root = frames_dir.parent

    # If output bases not provided, create top-level folders next to video_root
    # and keep per-video subfolders (consistent with explicit base behavior).
    if best_frame_output_dir is None or str(best_frame_output_dir).strip() == "":
        best_base = video_root.parent / "04_best_frame"
        best_frame_dir = best_base / video_root.name
    else:
        best_frame_dir = Path(best_frame_output_dir) / video_root.name

    if full_frames_output_dir is None or str(full_frames_output_dir).strip() == "":
        full_base = video_root.parent / "03_full_frames"
        full_frames_dir = full_base / video_root.name
    else:
        full_frames_dir = Path(full_frames_output_dir) / video_root.name

    # Ensure directories exist
    best_frame_dir.mkdir(parents=True, exist_ok=True)
    full_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best frame
    final_best_path = best_frame_dir / "best_frame.jpg"
    shutil.copy(best_frame, final_best_path)
    print(f"\n[✔] Best frame saved to: {final_best_path}")

    # Move full rack images to full_frames directory
    if full_rack_frames:
        print(f"[✔] Moving {len(full_rack_frames)} full rack images to: {full_frames_dir}")
        for frame_file, _ in tqdm(full_rack_frames, desc="Moving full rack frames"):
            dest = full_frames_dir / frame_file.name
            shutil.move(str(frame_file), dest)

    # Keep unit-wise frames in frames directory (already there)
    if unit_frames:
        print(f"[✔] Keeping {len(unit_frames)} unit-wise frames in: {frames_dir}")

    print("\n================ RESULT ================")
    print(f"[✔] Analyzed {len(frame_files)} frames")
    print(f"[✔] Best frame: {best_filename} (Sharpness: {best_score:.2f})")
    print(f"[✔] Moved {len(full_rack_frames)} full rack frames to: {full_frames_dir}")
    print(f"[✔] Kept {len(unit_frames)} unit-wise frames in: {frames_dir}")
    print("=======================================\n")


# ------------------ USAGE ------------------

# Provide only the frames directory; the script will create sibling
# `03_full_frames` and `04_best_frame` folders automatically.
frames_directory = r"C:\Users\SrikanthMekala\Downloads\video_frames\video_frames\grouped_frames_0850"
# Optionally override the defaults by providing explicit base folders:
# find_best_frame(frames_directory, r"C:\path\to\best_base", r"C:\path\to\full_base")

find_best_frame(frames_directory)
