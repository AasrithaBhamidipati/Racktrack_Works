#!/usr/bin/env python3
"""
Video Processing Pipeline
Usage: python main.py <video_path_or_folder> [--parallel]
Queue: 1→2→3→4 (validate→extract→analyze→divide)
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import cv2
import numpy as np
from collections import defaultdict
import shutil
from tqdm import tqdm
import os

SUPPORTED_FORMATS = [".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv"]

def find_ffmpeg():
    """Find FFmpeg executable in system PATH or common locations"""
    # Try to find in PATH
    result = subprocess.run(['where', 'ffmpeg.exe'], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip().split('\n')[0]
    
    # Common Windows installation paths
    common_paths = [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
        os.path.expandvars(r'%APPDATA%\..\Local\Programs\ffmpeg\bin\ffmpeg.exe'),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # If not found, raise an error
    raise FileNotFoundError(
        "FFmpeg not found. Please install FFmpeg from https://ffmpeg.org/download.html "
        "and add it to your system PATH, or download the portable version and extract to C:\\ffmpeg"
    )

def process_single_video(video_path_str):
    """Process a single video file (called by worker processes)"""
    video_path = Path(video_path_str)
    
    # Create output structure
    output_base = Path(__file__).parent / "output"
    video_name = video_path.stem
    video_output = output_base / video_name
    
    paths = {
        "validation": video_output / "01_validation",
        "frames": video_output / "02_frames",
        "full_frames": video_output / "03_full_frames",
        "best_frame": video_output / "04_best_frame",
        "units": video_output / "05_units",
        "logs": video_output / "logs",
    }
    
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PIPELINE: {video_path.name}")
    print(f"Output:   {video_output}")
    print(f"{'='*70}\n")
    
    # ===== STEP 1: EXTRACT FRAMES =====
    print("2/4 EXTRACT FRAMES...")
    try:
        ffmpeg_path = find_ffmpeg()
        frame_pattern = str(paths["frames"] / "frame_%010d.png")
        cmd = [
            ffmpeg_path, "-i", str(video_path),
            "-vf", "fps=1",
            "-c:v", "png", "-pix_fmt", "rgb24",
            "-compression_level", "9", "-y",
            frame_pattern
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        
        frame_count = len(list(paths["frames"].glob("frame_*.png")))
        if frame_count == 0:
            raise Exception("FFmpeg did not extract any frames. Check video file.")
        print(f"OK ({frame_count} frames)\n")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # ===== STEP 2: ANALYZE RACKS =====
    print("2/3 ANALYZE RACKS...")
    try:
        def calc_sharpness(img_path):
            if not Path(img_path).exists():
                return 0
            img = cv2.imread(str(img_path))
            if img is None:
                return 0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        
        frames = sorted(paths["frames"].glob("frame_*.png"))
        best_frame = None
        best_score = 0
        full_count = 0
        
        for frame in tqdm(frames, desc="  Analyzing frames for best/full", leave=False):
            if not frame.exists():
                continue
            score = calc_sharpness(str(frame))
            if score > best_score:
                best_score = score
                best_frame = frame
            if score > 50:
                full_count += 1
                try:
                    shutil.move(str(frame), paths["full_frames"] / frame.name)
                except Exception as move_err:
                    print(f"  Warning: Could not move {frame.name}: {move_err}")
        
        # Copy best frame to best_frame folder (if it still exists in 02_frames)
        if best_frame and best_frame.exists():
            try:
                shutil.copy(str(best_frame), paths["best_frame"] / "best_frame.jpg")
            except Exception as copy_err:
                print(f"  Warning: Could not copy best frame: {copy_err}")
        elif best_frame:
            # If best frame was moved to full_frames, copy from there
            full_path = paths["full_frames"] / best_frame.name
            if full_path.exists():
                try:
                    shutil.copy(str(full_path), paths["best_frame"] / "best_frame.jpg")
                except Exception as copy_err:
                    print(f"  Warning: Could not copy best frame: {copy_err}")
        
        print(f"OK (best:{best_frame.name if best_frame else 'N/A'}, full:{full_count})\n")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # ===== STEP 3: DIVIDE UNITS =====
    print("3/3 DIVIDE UNITS...")
    try:
        # Collect remaining frames (those not in full_frames due to low sharpness)
        frames = sorted(paths["frames"].glob("frame_*.png"))
        
        if not frames:
            print("OK (0 units - no remaining frames after filtering)\n")
        else:
            def get_edges(img_path):
                img = cv2.imread(str(img_path))
                if img is None:
                    return 0
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                return np.count_nonzero(edges) / edges.size
            
            densities = [get_edges(f) for f in frames]
            densities = np.convolve(densities, np.ones(3)/3, mode='same').tolist()
            
            changes = []
            for i in tqdm(range(1, len(densities)), desc="  Computing frame changes", leave=False):
                ch = abs(densities[i] - densities[i-1]) / max(densities[i-1], 0.001)
                changes.append(ch)
            
            transitions = []
            if changes:
                thresh = np.percentile(changes, 85)
                for i in range(1, len(changes)):
                    if changes[i] > thresh and changes[i] > changes[i-1]:
                        transitions.append(i)
            
            filtered = []
            for t in transitions:
                if not filtered or t - filtered[-1] > 10:
                    filtered.append(t)
            transitions = filtered
            
            units = defaultdict(list)
            current = 0
            for i, frame in enumerate(frames):
                if current < len(transitions) and i >= transitions[current]:
                    current += 1
                units[current].append(frame)
            
            for uid, uframes in tqdm(sorted(units.items()), desc="  Creating unit directories", leave=False):
                udir = paths["units"] / f"unit_{uid:02d}"
                udir.mkdir(parents=True, exist_ok=True)
                for f in uframes:
                    shutil.copy(str(f), udir / f.name)
            
            print(f"OK ({len(units)} units)\n")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    print("="*70)
    print("COMPLETE")
    print("="*70)
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path_or_folder> [--parallel]")
        print("Examples:")
        print("  python main.py C:\\videos\\rack.mp4              # Single video")
        print("  python main.py C:\\videos\\                      # All videos in folder")
        print("  python main.py C:\\videos\\ --parallel           # Parallel processing")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    use_parallel = "--parallel" in sys.argv
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    # Get list of videos to process
    videos = []
    
    if input_path.is_file():
        # Single video file
        if input_path.suffix.lower() not in SUPPORTED_FORMATS:
            print(f"Error: Unsupported format: {input_path.suffix}")
            sys.exit(1)
        videos = [input_path]
    else:
        # Folder - find all videos
        videos = sorted([
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
        ])
        
        if not videos:
            print(f"Error: No video files found in {input_path}")
            sys.exit(1)
    
    print(f"\nFound {len(videos)} video(s) to process")
    print(f"Parallel mode: {'ENABLED' if use_parallel else 'DISABLED'}\n")
    
    # Process videos
    if use_parallel and len(videos) > 1:
        # Parallel processing
        max_workers = min(4, len(videos))  # Max 4 parallel processes
        print(f"Starting {max_workers} parallel workers...\n")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_video, str(video)): video
                for video in videos
            }
            
            completed = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                video = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    status = "[OK]" if result else "[FAIL]"
                    tqdm.write(f"[{completed}/{len(videos)}] {status} {video.name}")
                except Exception as e:
                    tqdm.write(f"[{completed}/{len(videos)}] [FAIL] {video.name} - {e}")
                    completed += 1
    else:
        # Sequential processing
        for idx, video in tqdm(enumerate(videos, 1), total=len(videos), desc="Processing videos"):
            tqdm.write(f"[{idx}/{len(videos)}] Processing {video.name}...")
            try:
                result = process_single_video(str(video))
                status = "[OK]" if result else "[FAIL]"
                tqdm.write(f"{status} Completed")
            except Exception as e:
                tqdm.write(f"[FAIL] Failed - {e}")
    
    print(f"\n{'='*70}")
    print(f"All videos processed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
