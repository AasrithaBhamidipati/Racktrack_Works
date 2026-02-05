import os
import sys
import cv2
import base64
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

# ---------------------------
# Configuration
# ---------------------------
INPUT_FOLDER = r"D:\led\0905_units"
MODEL_PATH = r"D:\led\led_8s.pt"
OUTPUT_HTML = r"D:\led\unified_detection_report.html"
CONFIDENCE_THRESHOLD = 0.4
LED_DETECTION_THRESHOLD = 0.30

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


def image_to_base64(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error converting image: {e}")
        return None


def detect_leds(image_path, model):
    """Detect LEDs in image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        results = model.predict(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]
        
        original_b64 = image_to_base64(image_path)
        
        annotated_img = result.plot()
        annotated_path = Path(image_path).parent / f"temp_annotated_{Path(image_path).name}"
        cv2.imwrite(str(annotated_path), annotated_img)
        annotated_b64 = image_to_base64(annotated_path)
        
        if os.path.exists(annotated_path):
            os.remove(annotated_path)
        
        detections = len(result.boxes)
        if detections > 0:
            confidence = float(result.boxes.conf.max().cpu().numpy())
        else:
            confidence = 0.0
        
        return {
            'original_image': original_b64,
            'annotated_image': annotated_b64,
            'detections': detections,
            'class': "LED" if detections > 0 else "No LED",
            'confidence': f"{confidence:.1f}%",
            'filename': Path(image_path).name
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def scan_folders(input_folder):
    """Scan folders for images"""
    folder_structure = defaultdict(list)
    input_path = Path(input_folder)
    
    if not input_path.exists():
        return folder_structure
    
    for subfolder in sorted(input_path.iterdir()):
        if subfolder.is_dir():
            images = []
            for ext in IMAGE_EXTENSIONS:
                images.extend(sorted(subfolder.glob(f"*{ext}")))
                images.extend(sorted(subfolder.glob(f"*{ext.upper()}")))
            
            if images:
                images = sorted(set(images))
                folder_structure[subfolder.name] = images
    
    return folder_structure


def generate_html_report(results_by_folder, output_path):
    """Generate simple HTML report without buttons"""
    
    # Calculate stats
    total_images = sum(len(results) for results in results_by_folder.values())
    total_detections = sum(sum(1 for r in results if r['detections'] > 0) for results in results_by_folder.values())
    
    detected_confidences = []
    for results in results_by_folder.values():
        for result in results:
            if result['detections'] > 0:
                detected_confidences.append(float(result['confidence'].strip('%')))
    
    avg_confidence = sum(detected_confidences) / len(detected_confidences) if detected_confidences else 0
    
    # Classify folders
    switches = 0
    patch_panels = 0
    folder_info = {}
    
    for folder_name in sorted(results_by_folder.keys()):
        results = results_by_folder[folder_name]
        folder_detections = sum(1 for r in results if r['detections'] > 0)
        detection_rate = folder_detections / len(results) if results else 0
        
        if detection_rate >= LED_DETECTION_THRESHOLD:
            folder_type = "switch"
            switches += 1
        else:
            folder_type = "patch"
            patch_panels += 1
        
        folder_info[folder_name] = {
            'type': folder_type,
            'count': folder_detections,
            'total': len(results),
            'rate': detection_rate,
            'results': results
        }
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LED Detection Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            padding: 30px;
            background: #f9f9f9;
            border-bottom: 2px solid #ddd;
        }}
        
        .stat-box {{
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .info-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 4px;
            line-height: 1.8;
        }}
        
        .units-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .unit-card {{
            border: 3px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .unit-card.switch {{ border-color: #28a745; }}
        .unit-card.patch {{ border-color: #ffc107; }}
        
        .unit-header {{
            padding: 15px;
            color: white;
            font-size: 1.2em;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .unit-header.switch {{ background: linear-gradient(135deg, #28a745 0%, #20c997 100%); }}
        .unit-header.patch {{ background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); }}
        
        .unit-body {{
            padding: 20px;
        }}
        
        .unit-stat {{
            display: flex;
            justify-content: space-between;
            margin: 12px 0;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .unit-label {{ color: #666; font-weight: bold; }}
        .unit-value {{ color: #333; font-weight: bold; }}
        
        .frames-section {{
            margin-top: 40px;
            padding-top: 30px;
            border-top: 2px solid #ddd;
        }}
        
        .frame-box {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        
        .frame-header {{
            background: #f5f5f5;
            padding: 15px;
            font-weight: bold;
            border-bottom: 1px solid #ddd;
        }}
        
        .frame-content {{
            display: flex;
            height: 250px;
        }}
        
        .frame-img {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f9f9f9;
            position: relative;
        }}
        
        .frame-img img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        
        .frame-label {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(102, 126, 234, 0.9);
            color: white;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        
        .divider {{ width: 2px; background: #ddd; }}
        
        .frame-details {{
            padding: 15px;
            background: #f9f9f9;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            font-size: 0.9em;
        }}
        
        .detail-item {{
            display: flex;
            justify-content: space-between;
        }}
        
        .detail-label {{ color: #667eea; font-weight: bold; }}
        .detail-value {{ color: #333; }}
        
        .detected {{ color: #28a745; font-weight: bold; }}
        .not-detected {{ color: #dc3545; font-weight: bold; }}
        
        .footer {{
            background: #f9f9f9;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ LED Detection Report</h1>
            <p>Switch vs Patch Panel Analysis</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{len(results_by_folder)}</div>
                <div class="stat-label">Total Units</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #28a745;">{switches}</div>
                <div class="stat-label">🟢 Switches</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #ffc107;">{patch_panels}</div>
                <div class="stat-label">🟡 Patch Panels</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{total_images}</div>
                <div class="stat-label">Total Frames</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" style="color: #28a745;">{total_detections}</div>
                <div class="stat-label">LED Found</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{avg_confidence:.1f}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        </div>
        
        <div class="content">
            <div class="info-box">
                <h3>📊 How Classification Works</h3>
                <p><strong>Each folder = one unit from video</strong></p>
                <p style="margin-top: 10px;">
                    <strong style="color: #1565c0;">🟢 SWITCH:</strong> If <strong>more than 30%</strong> of frames show LED detection → This unit HAS LED
                </p>
                <p style="margin-top: 8px;">
                    <strong style="color: #1565c0;">🟡 PATCH PANEL:</strong> If <strong>30% or less</strong> of frames show LED detection → This unit has NO LED
                </p>
            </div>
            
            <div class="section-title">📁 All Units Summary</div>
            
            <div class="units-grid">
"""
    
    # Add unit cards
    for folder_name in sorted(folder_info.keys()):
        info = folder_info[folder_name]
        unit_type = info['type']
        count = info['count']
        total = info['total']
        rate = (count / total * 100) if total > 0 else 0
        
        type_label = "🟢 SWITCH" if unit_type == "switch" else "🟡 PATCH PANEL"
        
        html += f"""                <div class="unit-card {unit_type}">
                    <div class="unit-header {unit_type}">
                        <div>{folder_name}</div>
                        <div style="font-size: 0.8em;">{type_label}</div>
                    </div>
                    <div class="unit-body">
                        <div class="unit-stat">
                            <span class="unit-label">Total Frames:</span>
                            <span class="unit-value">{total}</span>
                        </div>
                        <div class="unit-stat">
                            <span class="unit-label">LED Found:</span>
                            <span class="unit-value">{count}</span>
                        </div>
                        <div class="unit-stat">
                            <span class="unit-label">Detection Rate:</span>
                            <span class="unit-value">{rate:.1f}%</span>
                        </div>
                    </div>
                </div>
"""
    
    html += """            </div>
            
            <div class="frames-section">
                <div class="section-title">📸 Detailed Frame Analysis</div>
"""
    
    # Add frame details for each unit
    for folder_name in sorted(folder_info.keys()):
        info = folder_info[folder_name]
        unit_type = info['type']
        type_label = "🟢 SWITCH" if unit_type == "switch" else "🟡 PATCH PANEL"
        
        html += f"""                <h3 style="margin-top: 30px; margin-bottom: 15px; color: #333; font-size: 1.2em;">
                    {folder_name} ({type_label})
                </h3>
"""
        
        for idx, result in enumerate(info['results'], 1):
            if not result or not result['original_image']:
                continue
            
            status = "✓ LED Detected" if result['detections'] > 0 else "✗ No LED"
            status_class = "detected" if result['detections'] > 0 else "not-detected"
            
            html += f"""                <div class="frame-box">
                    <div class="frame-header">
                        Frame {idx}: {result['filename']}
                        <span style="float: right; color: {'#28a745' if result['detections'] > 0 else '#dc3545'};">
                            {status}
                        </span>
                    </div>
                    <div class="frame-content">
                        <div class="frame-img">
                            <img src="data:image/png;base64,{result['original_image']}" alt="Original">
                            <span class="frame-label">Original</span>
                        </div>
                        <div class="divider"></div>
                        <div class="frame-img">
                            <img src="data:image/png;base64,{result['annotated_image']}" alt="Detection">
                            <span class="frame-label">Detection</span>
                        </div>
                    </div>
                    <div class="frame-details">
                        <div class="detail-item">
                            <span class="detail-label">Status:</span>
                            <span class="detail-value {status_class}">{status}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Detections:</span>
                            <span class="detail-value">{result['detections']} found</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Confidence:</span>
                            <span class="detail-value">{result['confidence']}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Class:</span>
                            <span class="detail-value">{result['class']}</span>
                        </div>
                    </div>
                </div>
"""
    
    html += """            </div>
        </div>
        
        <div class="footer">
            <p>⚡ LED Detection System | YOLOv8 Model</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Threshold: 30% of frames with LED detection = SWITCH<br>
                No buttons - Just scroll to view all units and their frames
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ Report generated: {output_path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print("🚀 Starting LED Detection Report Generation")
    print(f"Input: {INPUT_FOLDER}")
    print(f"Model: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found")
        return
    
    print("📦 Loading model...")
    try:
        model = YOLO(MODEL_PATH)
        print("✓ Model loaded")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("🔍 Scanning folders...")
    folder_structure = scan_folders(INPUT_FOLDER)
    
    if not folder_structure:
        print(f"❌ No images found")
        return
    
    print(f"✓ Found {len(folder_structure)} folders")
    
    results_by_folder = {}
    total_images = sum(len(images) for images in folder_structure.values())
    processed = 0
    
    for folder_name in sorted(folder_structure.keys()):
        print(f"  Processing {folder_name}...", end=" ")
        results = []
        
        for image_path in folder_structure[folder_name]:
            result = detect_leds(image_path, model)
            if result is not None:
                results.append(result)
            processed += 1
        
        if results:
            results_by_folder[folder_name] = results
            print(f"✓ ({len(results)} frames)")
    
    print("\n📝 Generating HTML...")
    if generate_html_report(results_by_folder, OUTPUT_HTML):
        print(f"✅ Done! Report: {OUTPUT_HTML}")
        print(f"\nSummary: {len(results_by_folder)} units, {processed} frames")
        for folder_name in sorted(results_by_folder.keys()):
            det = sum(1 for r in results_by_folder[folder_name] if r['detections'] > 0)
            print(f"  - {folder_name}: {len(results_by_folder[folder_name])} frames, {det} with LED")
    else:
        print("❌ Failed to generate report")


if __name__ == "__main__":
    main()
