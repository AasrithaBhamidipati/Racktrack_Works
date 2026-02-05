# Video Processing Pipeline

A complete end-to-end video processing pipeline that validates, extracts frames, analyzes rack quality, and organizes frames by operational units.

## Overview

This pipeline automates the entire workflow for processing datacenter rack videos:

1. **Validation** - Ensures video meets quality standards (resolution, FPS, lighting, motion)
2. **Frame Extraction** - Extracts frames from the video at 1 FPS
3. **Rack Analysis** - Identifies the best frame and full-rack images
4. **Unit Division** - Organizes remaining frames into operational unit subfolders

## Usage

### Basic Command Line Usage

```bash
python main.py <video_path>
```

### Examples

```bash
# Using absolute path
python main.py "D:\videos\rack_scan_001.mp4"

# Using relative path
python main.py videos\rack_scan_001.mp4

# Interactive mode (no arguments)
python main.py
# Then enter the video path when prompted
```

### Supported Video Formats

- `.mp4` - H.264/H.265 (recommended)
- `.mov` - QuickTime format
- `.mkv` - Matroska format
- `.avi` - AVI format
- `.flv` - Flash Video
- `.wmv` - Windows Media Video

## Video Quality Requirements

For a video to be accepted by the pipeline, it must meet these standards:

### Resolution & Frame Rate
- **Minimum Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 24-120 FPS
- **Recommended FPS**: 30-60 FPS

### Video Quality
- **Lighting**: Well-lit with even exposure (not too dark or bright)
- **Focus**: Frames should be sharp and in focus
- **Blur**: Less than 30% of frames can be blurry

### Motion & Camera Movement
- **Motion Type**: Vertical panning (scanning from top to bottom)
- **Motion Speed**: 5-30 pixels per frame (optimal)
  - Too slow (<5 px/frame): Cannot see details
  - Too fast (>30 px/frame): Frames become unreadable
- **Motion Consistency**: Smooth, steady movement (not jerky)

### Rack Coverage
- **Vertical Travel**: Must cover at least 35% of the frame height
- **Rack Visibility**: Rack edges visible in at least 90% of frames
- **Occlusion**: No more than 20% occlusion in any frame

### Port/Component Visibility
- **Zoom Level**: Must show individual ports/components clearly
- **Minimum Objects**: At least 8 port-like objects must be detectable
- **Separation**: Ports must be at least 10 pixels apart

## Output Directory Structure

After successful processing, your video output will be organized as:

```
output/
└── <video_name>/
    ├── 01_validation/
    │   └── validation_report.html
    ├── 02_frames/
    │   ├── frame_0000000001.png
    │   ├── frame_0000000002.png
    │   └── ... (all extracted frames)
    ├── 03_full_frames/
    │   ├── frame_xxxx.png
    │   └── frame_yyyy.png
    │   └── ... (high-quality full rack frames)
    ├── 04_best_frame/
    │   └── best_frame.jpg
    │   └── ... (sharpest single frame)
    ├── 05_units/
    │   ├── unit_00/
    │   │   ├── frame_xxx.png
    │   │   └── frame_xxx.png
    │   ├── unit_01/
    │   │   ├── frame_xxx.png
    │   │   └── frame_xxx.png
    │   └── ... (additional units)
    └── logs/
        └── pipeline.log
```

## Folder Descriptions

| Folder | Description |
|--------|-------------|
| `01_validation/` | Validation report and metadata |
| `02_frames/` | All extracted frames (1 FPS) |
| `03_full_frames/` | Full-rack images (sharpness > 50) |
| `04_best_frame/` | Single sharpest frame |
| `05_units/` | Frames grouped by operational units |
| `logs/` | Processing logs and debug info |

## Step-by-Step Workflow

### Step 1: Video Validation
- Extracts video metadata (resolution, FPS, codec, bitrate)
- Analyzes video quality (blur, sharpness, lighting)
- Detects rack scanning motion pattern
- Verifies port/component visibility
- Checks rack overview coverage

**If validation fails**: Pipeline stops and provides specific improvement guidelines.

### Step 2: Frame Extraction
- Extracts 1 frame per second from the video
- Saves frames as PNG files (lossless)
- All frames are preserved for analysis

### Step 3: Rack Analysis
- Calculates sharpness score for each frame
- Separates **full-rack frames** (high sharpness > 50)
- Identifies **best frame** (sharpest single frame)
- Keeps remaining frames for unit-level analysis

### Step 4: Unit Division
- Analyzes frame-to-frame transitions
- Detects unit boundaries based on zoom changes
- Organizes frames into unit subfolders (unit_00, unit_01, etc.)
- Each unit contains all frames showing that specific rack unit

## Requirements

### Python Packages
```bash
pip install opencv-python numpy pymediainfo
```

### System Dependencies
- **FFmpeg**: Required for video processing
  - Windows: Download from https://ffmpeg.org/download.html
  - Add FFmpeg to system PATH or ensure `ffmpeg` command is available

### Disk Space
- **Required**: Approximately 5-10x the video file size
  - Video file: ~100 MB
  - Output frames: ~500-1000 MB (PNG format)
  - Total: ~600-1100 MB

## Troubleshooting

### Error: "FFmpeg not found"
**Solution**: Install FFmpeg and add it to your system PATH
- Windows: https://ffmpeg.org/download.html

### Error: "Video file not found"
**Solution**: Check the file path is correct and file exists
```bash
# Verify file exists
Test-Path "D:\videos\my_video.mp4"
```

### Error: "pymediainfo not found"
**Solution**: Install the package
```bash
pip install pymediainfo
```

### Video Rejected: Resolution Below 1080p
**Solution**: Re-record the video with a higher resolution camera/phone
- Most modern smartphones: 1920x1080 minimum
- Ensure camera is in landscape mode

### Video Rejected: Motion Too Fast
**Solution**: Move the camera more slowly when scanning the rack
- Optimal speed: 10-20 pixels per frame
- Take about 30-60 seconds to scan entire rack

### Video Rejected: Frames Too Blurry
**Solution**: Ensure proper lighting and steady camera movement
- Use additional lighting if needed
- Ensure camera autofocus is working
- Avoid sudden movements

## Pipeline Execution Example

```
python main.py "C:\videos\datacenter_rack_001.mp4"

######################################################################
# VIDEO PROCESSING PIPELINE
#=====================================================================#
# Input: datacenter_rack_001.mp4
# Started: 2026-01-28 15:45:30
#=====================================================================#

======================================================================
STEP 1: VIDEO VALIDATION
======================================================================
[→] Running validation checks...

✓ VIDEO VALIDATION PASSED
  - AC1 Metadata: PASS
  - AC2 Quality: PASS (Score: 92/100)
  - AC3 Rack Scan: PASS
  - AC5 Ports: PASS
  - Rack Overview: PASS

======================================================================
STEP 2: FRAME EXTRACTION
======================================================================
[→] Extracting frames from video...

✓ FRAME EXTRACTION COMPLETED
  - Total frames: 287
  - Output: Z:\Video_Test\output\datacenter_rack_001\02_frames

======================================================================
STEP 3: FULL RACK ANALYSIS
======================================================================
[→] Analyzing frames for rack quality...

✓ RACK ANALYSIS COMPLETED
  - Best frame saved: frame_0000000145.png (Sharpness: 245.67)
  - Full rack frames: 142
  - Remaining frames (for unit division): 145
  - Output: Z:\Video_Test\output\datacenter_rack_001\03_full_frames

======================================================================
STEP 4: UNIT FRAME DIVISION
======================================================================
[→] Dividing frames into unit subfolders...

✓ UNIT DIVISION COMPLETED
  - Units created: 5
  - Total frames in units: 145
  - Output: Z:\Video_Test\output\datacenter_rack_001\05_units

  Unit Summary:
    ├─ unit_00: 34 frames
    ├─ unit_01: 28 frames
    ├─ unit_02: 31 frames
    ├─ unit_03: 26 frames
    └─ unit_04: 26 frames

======================================================================
✓ PIPELINE COMPLETED SUCCESSFULLY
======================================================================

Output Location: Z:\Video_Test\output\datacenter_rack_001

Folder Structure:
  ├── 01_validation/    (Validation report)
  ├── 02_frames/        (All extracted frames)
  ├── 03_full_frames/   (Full rack images)
  ├── 04_best_frame/    (Best quality frame)
  ├── 05_units/         (Frames organized by unit)
  └── logs/             (Processing logs)
```

## Tips for Best Results

1. **Lighting**: Ensure the datacenter is well-lit but without harsh shadows
2. **Camera Stability**: Use a tripod or stabilizer for smooth motion
3. **Zoom Level**: Start zoomed in on ports, then zoom out for overview
4. **Speed**: Scan at a moderate pace (1-2 seconds per unit)
5. **Completeness**: Capture the entire rack from top to bottom
6. **Video Format**: Use H.264 codec in MP4 container for best compatibility
7. **Bitrate**: Use at least 50 Mbps for good quality

## Common Validation Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Resolution too low | Camera quality or settings | Use 1080p+ camera |
| Frames too blurry | Poor lighting or focus | Improve lighting, check focus |
| Motion too fast | Scanning too quickly | Slow down the scan |
| Motion too slow | Scanning too slowly | Speed up the scan |
| Jerky movement | Shaky camera | Use tripod/stabilizer |
| Incomplete rack | Not scanning full height | Ensure top-to-bottom coverage |
| Poor port visibility | Insufficient zoom | Zoom in on individual ports |

## Advanced Configuration

Edit `validate_video.py` to adjust thresholds:

```python
CONFIG = {
    "min_width": 1920,          # Minimum resolution width
    "min_height": 1080,         # Minimum resolution height
    "min_fps": 24,              # Minimum frame rate
    "max_fps": 120,             # Maximum frame rate
    # ... more settings
}
```

## Support

For issues or questions:
1. Check the validation report in `01_validation/`
2. Review logs in the `logs/` folder
3. Verify all system requirements are installed
4. Ensure video meets quality standards
