import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import base64
from datetime import datetime

# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = r"D:\led\Led_Detection.v1i.yolov8\LED_Detection\YOLOv8m_LED\weights"
THRESHOLDS = [0.3, 0.4, 0.5]

# ---------------------------
# Supported image formats
# ---------------------------
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


def image_to_base64(image_path):
    """Convert image to base64 string for embedding in HTML"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode()


def detect_leds(image_path, model, confidence_threshold):
    """
    Detect LEDs in image and return results
    
    Returns:
        dict: {
            'original_image': base64 encoded original image,
            'annotated_image': base64 encoded annotated image,
            'detections': number of LEDs detected,
            'class': 'switch' if LED detected else 'patch panel',
            'confidence': max confidence score (if detected)
        }
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Run inference
    results = model.predict(image_path, conf=confidence_threshold, verbose=False)
    result = results[0]
    
    # Create annotated image
    annotated_img = result.plot()
    annotated_path = image_path.parent / f"temp_annotated_{image_path.name}"
    cv2.imwrite(str(annotated_path), annotated_img)
    annotated_b64 = image_to_base64(annotated_path)
    os.remove(annotated_path)
    
    # Extract detection information
    detections = len(result.boxes)
    
    if detections > 0:
        # LED detected
        class_label = "switch"
        confidence = float(result.boxes.conf.max().cpu().numpy())
    else:
        # No LED detected
        class_label = "patch panel"
        confidence = 0.0
    
    return {
        'annotated_image': annotated_b64,
        'detections': detections,
        'class': class_label,
        'confidence': f"{confidence:.2%}"
    }


def generate_html_report(results_by_threshold, original_images, output_path):
    """Generate HTML report with test results for multiple thresholds"""
    
    # Get all image filenames (same across thresholds)
    image_filenames = sorted(list(original_images.keys()))
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LED Detection Multi-Threshold Results</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f5f5f5;
                padding: 20px;
                color: #333;
            }
            
            .container {
                max-width: 1800px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.2em;
                margin-bottom: 10px;
                font-weight: 700;
            }
            
            .header p {
                font-size: 1em;
                opacity: 0.95;
            }
            
            .stats-section {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f9f9f9;
                border-bottom: 1px solid #e0e0e0;
            }
            
            .stat-box {
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #667eea;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            
            .stat-box h3 {
                font-size: 2.5em;
                color: #667eea;
                margin-bottom: 8px;
            }
            
            .stat-box p {
                color: #666;
                font-size: 0.95em;
            }
            
            .content {
                padding: 30px;
            }
            
            .comparison-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }
            
            .comparison-table th {
                background: #667eea;
                color: white;
                padding: 15px;
                text-align: center;
                font-weight: 600;
                border: 1px solid #e0e0e0;
                font-size: 0.95em;
            }
            
            .comparison-table td {
                border: 1px solid #e0e0e0;
                padding: 15px;
                vertical-align: middle;
                background: white;
            }
            
            .comparison-table tr:nth-child(even) {
                background: #fafafa;
            }
            
            .comparison-table tr:hover {
                background: #f0f8ff;
            }
            
            .original-image-cell {
                text-align: center;
                width: 120px;
            }
            
            .original-image-cell img {
                width: 100px;
                height: 100px;
                object-fit: cover;
                border-radius: 4px;
                border: 1px solid #ddd;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .original-image-cell img:hover {
                transform: scale(1.1);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
            
            .image-filename {
                font-size: 0.8em;
                color: #999;
                margin-top: 8px;
                word-break: break-word;
                max-width: 110px;
            }
            
            .threshold-cell {
                text-align: center;
                width: 140px;
            }
            
            .detection-box {
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 10px;
            }
            
            .detection-image {
                width: 100px;
                height: 100px;
                object-fit: cover;
                border-radius: 4px;
                margin-bottom: 8px;
                cursor: pointer;
                border: 1px solid #ddd;
                transition: all 0.3s ease;
            }
            
            .detection-image:hover {
                transform: scale(1.05);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
            }
            
            .detection-stats {
                font-size: 0.85em;
                line-height: 1.6;
            }
            
            .led-count {
                font-weight: 700;
                font-size: 1.3em;
                color: #667eea;
                margin: 5px 0;
            }
            
            .class-badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 0.75em;
                font-weight: 600;
                margin: 5px 0;
            }
            
            .badge-switch {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .badge-panel {
                background: #cfe2ff;
                color: #084298;
                border: 1px solid #b6d4fe;
            }
            
            .confidence-text {
                color: #666;
                font-size: 0.8em;
            }
            
            .threshold-header {
                font-weight: 700;
                color: #667eea;
                font-size: 0.95em;
                margin-bottom: 8px;
            }
            
            .footer {
                background: #f9f9f9;
                padding: 25px;
                text-align: center;
                color: #666;
                border-top: 1px solid #e0e0e0;
                font-size: 0.9em;
            }
            
            .image-viewer {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.8);
                animation: fadeIn 0.3s;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            .image-viewer-content {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 90%;
                max-width: 900px;
                max-height: 90vh;
            }
            
            .image-viewer img {
                width: 100%;
                height: auto;
                border-radius: 5px;
            }
            
            .close-viewer {
                position: absolute;
                top: 20px;
                right: 30px;
                color: #fff;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
                z-index: 1001;
                transition: color 0.3s;
            }
            
            .close-viewer:hover {
                color: #bbb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 LED Detection Analysis</h1>
                <p>Comparing Detection Results Across Three Confidence Thresholds: 0.3 | 0.4 | 0.5</p>
            </div>
            
            <div class="stats-section">
    """
    
    # Calculate overall statistics
    results_0_3 = results_by_threshold[0.3]
    results_0_4 = results_by_threshold[0.4]
    results_0_5 = results_by_threshold[0.5]
    
    detected_0_3 = sum(1 for r in results_0_3 if r['detections'] > 0)
    detected_0_4 = sum(1 for r in results_0_4 if r['detections'] > 0)
    detected_0_5 = sum(1 for r in results_0_5 if r['detections'] > 0)
    
    total = len(results_0_4)
    
    html_content += f"""
                <div class="stat-box">
                    <h3>{total}</h3>
                    <p>Total Images Tested</p>
                </div>
                <div class="stat-box">
                    <h3>{detected_0_3}</h3>
                    <p>Detected @ 0.3</p>
                </div>
                <div class="stat-box">
                    <h3>{detected_0_4}</h3>
                    <p>Detected @ 0.4</p>
                </div>
                <div class="stat-box">
                    <h3>{detected_0_5}</h3>
                    <p>Detected @ 0.5</p>
                </div>
            </div>
            
            <div class="content">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Image</th>
                            <th colspan="1">Threshold: 0.3</th>
                            <th colspan="1">Threshold: 0.4</th>
                            <th colspan="1">Threshold: 0.5</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Create a mapping of filenames to results for easier lookup
    results_map = {}
    for threshold in THRESHOLDS:
        results_map[threshold] = {r['filename']: r for r in results_by_threshold[threshold]}
    
    # Generate table rows for each image
    for filename in image_filenames:
        html_content += f"""
                        <tr>
                            <td class="original-image-cell">
                                <img src="data:image/jpeg;base64,{original_images[filename]}" alt="{filename}" title="Click to enlarge">
                                <div class="image-filename">{filename}</div>
                            </td>
    """
        
        # Add results for each threshold
        for threshold in THRESHOLDS:
            result = results_map[threshold].get(filename)
            if result:
                class_css = "badge-switch" if result['class'] == "switch" else "badge-panel"
                html_content += f"""
                            <td class="threshold-cell">
                                <div class="detection-box">
                                    <div class="threshold-header">{threshold}</div>
                                    <img src="data:image/jpeg;base64,{result['annotated_image']}" alt="Detection at {threshold}" class="detection-image" title="Click to enlarge">
                                    <div class="detection-stats">
                                        <div class="led-count">{result['detections']}</div>
                                        <span class="class-badge {class_css}">{result['class'].upper()}</span>
                                        <div class="confidence-text">Conf: {result['confidence']}</div>
                                    </div>
                                </div>
                            </td>
    """
        
        html_content += """
                        </tr>
    """
    
    # Close table and add footer
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content += f"""
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p><strong>Report generated on {timestamp}</strong></p>
                <p>Model: yolov8s_led | Click on any image to enlarge</p>
            </div>
        </div>
        
        <div id="imageViewer" class="image-viewer">
            <span class="close-viewer">&times;</span>
            <div class="image-viewer-content">
                <img id="viewerImage" src="" alt="Enlarged Image">
            </div>
        </div>
        
        <script>
            const viewer = document.getElementById('imageViewer');
            const viewerImage = document.getElementById('viewerImage');
            const closeBtn = document.querySelector('.close-viewer');
            
            // Attach click listeners to all images
            document.querySelectorAll('img[title*="enlarge"]').forEach(img => {{
                img.addEventListener('click', function() {{
                    viewer.style.display = 'block';
                    viewerImage.src = this.src;
                }});
            }});
            
            document.querySelectorAll('.detection-image').forEach(img => {{
                img.addEventListener('click', function() {{
                    viewer.style.display = 'block';
                    viewerImage.src = this.src;
                }});
            }});
            
            closeBtn.addEventListener('click', function() {{
                viewer.style.display = 'none';
            }});
            
            viewer.addEventListener('click', function(e) {{
                if (e.target === viewer) {{
                    viewer.style.display = 'none';
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test LED Detection Model with Multiple Thresholds')
    parser.add_argument('input_folder', help='Path to folder containing test images')
    parser.add_argument('-o', '--output', default='led_detection_multi_threshold.html', 
                        help='Output HTML file path (default: led_detection_multi_threshold.html)')
    
    args = parser.parse_args()
    
    # Validate input folder
    input_path = Path(args.input_folder)
    if not input_path.exists() or not input_path.is_dir():
        print(f"❌ Error: Input folder '{args.input_folder}' does not exist!")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at '{MODEL_PATH}'")
        sys.exit(1)
    
    print(f"🚀 Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Get all image files (remove duplicates with set)
    image_files_set = set()
    for ext in IMAGE_EXTENSIONS:
        image_files_set.update(input_path.glob(f'*{ext}'))
        image_files_set.update(input_path.glob(f'*{ext.upper()}'))
    
    image_files = sorted(list(image_files_set))
    
    if not image_files:
        print(f"❌ No images found in '{args.input_folder}'")
        sys.exit(1)
    
    print(f"📸 Found {len(image_files)} images to process...")
    
    # Store original images
    original_images = {}
    for image_path in image_files:
        original_images[image_path.name] = image_to_base64(image_path)
    
    # Process images with each threshold
    results_by_threshold = {}
    
    for threshold in THRESHOLDS:
        print(f"\n🎯 Processing with threshold: {threshold}")
        results_list = []
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}...", end=' ')
            
            result = detect_leds(image_path, model, threshold)
            if result:
                result['filename'] = image_path.name
                result['original_image'] = original_images[image_path.name]
                results_list.append(result)
                print(f"✓ ({result['detections']} LED(s) - {result['class']})")
            else:
                print("❌ Failed to process image")
        
        results_by_threshold[threshold] = results_list
    
    # Generate HTML report
    if results_by_threshold:
        generate_html_report(results_by_threshold, original_images, args.output)
        print(f"\n📊 Summary:")
        for threshold in THRESHOLDS:
            results = results_by_threshold[threshold]
            detected = sum(1 for r in results if r['detections'] > 0)
            print(f"   Threshold {threshold}: {detected}/{len(results)} images with LEDs detected")
    else:
        print("❌ No images were processed successfully!")
        sys.exit(1)


if __name__ == '__main__':
    main()
