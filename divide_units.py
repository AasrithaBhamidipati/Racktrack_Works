import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default input folder containing subfolders with frames
INPUT_FOLDER = r"Z:\Video_Test\Videoframes"
OUTPUT_DIR = r"Z:\Video_Test\units_chatco"
ZOOM_OUT_PERCENTILE = 25  # Adjust this if zoom-out detection is wrong
VIDEOS_PER_BATCH = 8  # Process 4 videos at a time

def get_edge_density(image_path: Path) -> float:
    """Calculate edge density to detect zoom/focus changes"""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    return edge_density

def get_center_region_brightness(image_path: Path) -> float:
    """Get brightness of center region to detect zoom transitions"""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Get center 40% of image
    cy1, cy2 = int(h * 0.3), int(h * 0.7)
    cx1, cx2 = int(w * 0.3), int(w * 0.7)
    center = gray[cy1:cy2, cx1:cx2]
    return np.mean(center)

def detect_unit_transition(prev_edges: float, curr_edges: float, prev_brightness: float, curr_brightness: float) -> bool:
    """Detect unit transition by looking for significant changes in edge density and brightness"""
    edge_change = abs(curr_edges - prev_edges) / (max(prev_edges, 0.001))
    brightness_change = abs(curr_brightness - prev_brightness) / (max(prev_brightness, 1))
    
    # Unit transition happens when both edge density and brightness change significantly
    # This indicates zoom out to next unit
    return (edge_change > 0.5 and brightness_change > 0.15)

def smooth_signal(signal: list, window: int = 3) -> list:
    """Smooth signal using moving average"""
    if len(signal) < window:
        return signal
    smoothed = signal.copy()
    for i in range(window, len(signal)):
        smoothed[i] = np.mean(signal[i-window:i])
    return smoothed

def divide_frames_into_units(frames_dir: Path, video_name: str, base_output_dir: Path):
    """Process frames from a single video into units"""
    output_path = base_output_dir / video_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all frames sorted
    all_frames = sorted(frames_dir.glob("frame_*.png"))
    
    if not all_frames:
        logger.warning(f"No frames found in {frames_dir}")
        return False
    
    logger.info(f"[{video_name}] Found {len(all_frames)} total frames")
    
    # Calculate metrics for all frames
    logger.info(f"[{video_name}] Analyzing frames...")
    edge_densities = []
    brightnesses = []
    
    for frame_path in tqdm(all_frames, desc=f"  [{video_name}] Analyzing frames", leave=False):
        edge_density = get_edge_density(frame_path)
        brightness = get_center_region_brightness(frame_path)
        edge_densities.append(edge_density)
        brightnesses.append(brightness)
    
    # Smooth signals to reduce noise
    edge_densities = smooth_signal(edge_densities, window=3)
    brightnesses = smooth_signal(brightnesses, window=3)
    
    # Calculate changes between frames
    edge_changes = []
    brightness_changes = []
    combined_changes = []
    
    for i in tqdm(range(1, len(all_frames)), desc=f"  [{video_name}] Computing frame changes", leave=False):
        edge_change = abs(edge_densities[i] - edge_densities[i-1]) / (max(edge_densities[i-1], 0.001))
        brightness_change = abs(brightnesses[i] - brightnesses[i-1]) / (max(brightnesses[i-1], 1))
        edge_changes.append(edge_change)
        brightness_changes.append(brightness_change)
        combined_changes.append(edge_change * 0.5 + brightness_change * 0.5)
    
    # Find peaks in combined changes - these are likely unit transitions
    transitions = []
    threshold = np.percentile(combined_changes, 85)  # Top 15% of changes (should give us ~4-5 peaks)
    
    logger.info(f"[{video_name}] Change threshold: {threshold:.4f}")
    
    for i in range(1, len(combined_changes)):
        if combined_changes[i] > threshold and combined_changes[i] > combined_changes[i-1]:
            transitions.append(i)
            logger.info(f"[{video_name}] Potential transition at frame {i} (change: {combined_changes[i]:.4f}, edge: {edge_changes[i]:.4f}, brightness: {brightness_changes[i]:.4f})")
    
    # Remove duplicate transitions that are too close
    filtered_transitions = []
    for t in transitions:
        if not filtered_transitions or t - filtered_transitions[-1] > 10:
            filtered_transitions.append(t)
    
    # Adjust transitions to include zoom-out frames with current unit
    # Zoom-out typically happens a few frames before the major transition
    adjusted_transitions = []
    for t in filtered_transitions:
        # Keep original detected peaks (they already include some zoom-out context)
        if not adjusted_transitions or t - adjusted_transitions[-1] > 8:
            adjusted_transitions.append(t)
    
    transitions = adjusted_transitions
    
    logger.info(f"[{video_name}] Adjusted transitions: {transitions}")
    logger.info(f"[{video_name}] Detected {len(transitions) + 1} units")
    
    # Organize frames into units (all frames included)
    units = defaultdict(list)
    current_unit = 0
    
    for i, frame_path in enumerate(all_frames):
        if current_unit < len(transitions) and i >= transitions[current_unit]:
            current_unit += 1
        
        units[current_unit].append(frame_path)
    
    # Create unit directories with all frames
    for unit_id, frames in tqdm(sorted(units.items()), desc=f"  [{video_name}] Creating units", leave=False):
        unit_dir = output_path / f"unit_{unit_id:02d}"
        unit_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[{video_name}] Unit {unit_id}: {len(frames)} frames")
        
        for frame_path in frames:
            dest_path = unit_dir / frame_path.name
            import shutil
            shutil.copy(str(frame_path), str(dest_path))
    
    logger.info(f"[{video_name}] Frames divided into {len(units)} units")
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"VIDEO: {video_name}")
    print(f"{'='*50}")
    for unit_id in sorted(units.keys()):
        frames = units[unit_id]
        print(f"  Unit {unit_id}: {len(frames)} frames")
    print(f"{'='*50}\n")
    
    return True

def process_videos(input_folder: str = None, batch_size: int = VIDEOS_PER_BATCH):
    """Process multiple video folders in batches"""
    if input_folder is None:
        input_folder = INPUT_FOLDER
    
    input_path = Path(input_folder)
    
    if not input_path.exists():
        logger.error(f"Input folder not found: {input_folder}")
        return
    
    # Find all subdirectories (videos)
    video_folders = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    if not video_folders:
        logger.error(f"No subdirectories (videos) found in {input_folder}")
        return
    
    logger.info(f"Found {len(video_folders)} video folders")
    
    # Process videos in batches
    for batch_start in range(0, len(video_folders), batch_size):
        batch_end = min(batch_start + batch_size, len(video_folders))
        batch_videos = video_folders[batch_start:batch_end]
        
        batch_num = (batch_start // batch_size) + 1
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING BATCH {batch_num} ({len(batch_videos)} videos)")
        logger.info(f"{'='*60}\n")
        
        for video_folder in tqdm(batch_videos, desc=f"Processing batch {batch_num}"):
            video_name = video_folder.name
            logger.info(f"Processing: {video_name}")
            divide_frames_into_units(video_folder, video_name, Path(OUTPUT_DIR))
            logger.info(f"Completed: {video_name}\n")
        
        logger.info(f"Batch {batch_num} complete. Output in: {OUTPUT_DIR}\n")
        
        # Optional: Add pause between batches if needed
        if batch_end < len(video_folders):
            input(f"Press Enter to continue with the next batch ({batch_end}/{len(video_folders)} videos processed)...")

if __name__ == "__main__":
    # Get input folder from command line or use default
    input_folder = sys.argv[1] if len(sys.argv) > 1 else INPUT_FOLDER
    process_videos(input_folder)
