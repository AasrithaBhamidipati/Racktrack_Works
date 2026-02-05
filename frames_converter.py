import subprocess
from pathlib import Path
import logging
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIDEOS_DIR = r"Z:\Video_Test\videos"
OUTPUT_DIR = r"Z:\Video_Test\Videoframes"

FPS = 1  # 3 frames per second
BLUR_THRESHOLD = 100  # Laplacian variance threshold - only remove severely blurred
MAX_WORKERS = 15  # Process 4 videos in parallel

def is_heavily_blurred(image_path: Path, threshold=BLUR_THRESHOLD) -> bool:
    """Check if image is heavily blurred using Laplacian variance"""
    img = cv2.imread(str(image_path))
    if img is None:
        return True
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return variance < threshold

def extract_frames(video_path: Path):
    """Extract frames from a single video"""
    try:
        output_path = Path(OUTPUT_DIR) / video_path.stem
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing: {video_path.name}")

        frame_pattern = str(output_path / "frame_%010d.png")

        # Extract at specified FPS - no scene detection
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={FPS}",
            "-c:v", "png",
            "-pix_fmt", "rgb24",
            "-compression_level", "9",
            "-y",
            frame_pattern
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"[{video_path.name}] Error: {result.stderr}")
            return

        # Keep all frames - no blur filtering
        all_frames = sorted(output_path.glob("frame_*.png"))
        logger.info(f"[{video_path.name}] Extracted {len(all_frames)} total frames")
        logger.info(f"[{video_path.name}] Kept all {len(all_frames)} frames")
        
    except Exception as e:
        logger.error(f"Error processing {video_path.name}: {str(e)}")

def main():
    videos = sorted([
        v for v in Path(VIDEOS_DIR).glob("*")
        if v.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi")
    ])

    if not videos:
        logger.error(f"No videos found in {VIDEOS_DIR}")
        return

    logger.info(f"Found {len(videos)} video(s) to process")
    for v in videos:
        logger.info(f"  - {v.name}")

    logger.info(f"Starting parallel processing with {MAX_WORKERS} workers...")
    
    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(extract_frames, video) for video in videos]
        # Wait for all tasks to complete
        for future in tqdm(futures, desc="Extracting frames from videos"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Task failed: {str(e)}")

    logger.info("All videos processed successfully")

if __name__ == "__main__":
    main()
