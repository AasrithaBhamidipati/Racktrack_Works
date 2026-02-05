import torch
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm
import html
import re
import pandas as pd
import torch.nn.functional as F
import argparse

def main():
    # Parse arguments first
    parser = argparse.ArgumentParser(description="Patch panel classification")
    parser.add_argument("--input", type=str, default=None, help="Input folder with patch panel images")
    parser.add_argument("--output", type=str, default=None, help="Output folder for results")
    args = parser.parse_args()
    
    # Configuration with defaults
    test_dir = args.input if args.input else 'segmented_output/patch_panel'
    model_path = 'Trained_Models/patch_panel_identify.pth'
    output_html = os.path.join(args.output, "3_patchpanel_results.html") if args.output else 'Results/3_patchpanel_results.html'
    excel_file = 'Description_excel/PatchPanel_Descriptions.xlsx'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[3_patchpanel_match] test_dir={test_dir}")
    print(f"[3_patchpanel_match] output_html={output_html}")
    
    # Check if input directory exists
    if not os.path.exists(test_dir):
        print(f"[3_patchpanel_match] Input directory does not exist: {test_dir}")
        print(f"[3_patchpanel_match] Skipping patch panel classification - no images to process")
        # Create empty HTML report
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write('<html><head><title>Patch Panel Test Results</title></head><body>')
            f.write('<h1>Patch Panel Classification Results</h1>')
            f.write('<p>No patch panel images found to process.</p>')
            f.write('</body></html>')
        print(f"Empty report saved to {output_html}")
        return

    # Image transformations (must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Define classes
    class_names = [
        'BNC', 'DVI', 'FC', 'F_type', 'HDMI', 'IEC_C13_C14', 'IEC_C19_C20',
        'LC UPC', 'LC_APC', 'NEMA_5_15', 'RCA', 'RF_Coax', 'RJ_45', 'SC_APC',
        'SC_UPC', 'ST', 'TNC', 'USB_2_B', 'USB_3_B', 'USB_A', 'VGA', 'XLR', 'keystone'
    ]
    num_classes = len(class_names)

    # Load the model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Read Excel descriptions
    df_desc = pd.read_excel(excel_file)
    desc_dict = dict(zip(df_desc['Patch_Panel'], df_desc['Description']))

    # Gather test image paths
    image_paths = sorted(
        [os.path.join(test_dir, fname)
         for fname in os.listdir(test_dir)
         if fname.lower().endswith(('jpg', 'jpeg', 'png'))]
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    # Generate HTML
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write('<html><head><title>Patch Panel Test Results</title>\n')
        f.write('''
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .table-container { max-height: 90vh; overflow-y: auto; border: 1px solid #ccc; padding: 5px; }
            table { border-collapse: collapse; width: 100%; table-layout: fixed; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: center; word-wrap: break-word; }
            th { background-color: #f2f2f2; position: sticky; top: 0; z-index: 2; }
            tr:nth-child(even) { background-color: #fafafa; }
            img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
        </style>
        </head><body>
        <h1>Patch Panel Analysis Report</h1>
        <div class="table-container">
        <table>
            <tr>
                <th>Patch Panel</th>
                <th>Predicted Model</th>
                <th>Description</th>
                <th>Similarity</th>
            </tr>
        ''')

        for img_path in tqdm(image_paths, desc="Testing Images"):
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                predicted_class = class_names[pred.item()]
                confidence = conf.item()

            # Determine similarity
            if confidence >= 0.75:
                similarity = 'high'
            elif confidence >= 0.4:
                similarity = 'medium'
            else:
                similarity = 'low'

            description = desc_dict.get(predicted_class, '')

            img_relative_path = os.path.relpath(img_path, start=os.path.dirname(output_html)).replace('\\', '/')

            f.write('<tr>')
            f.write(f'<td><img src="{html.escape(img_relative_path)}" alt="{html.escape(predicted_class)}"></td>')
            f.write(f'<td>{html.escape(predicted_class)}</td>')
            f.write(f'<td>{html.escape(description)}</td>')
            f.write(f'<td>{similarity}</td>')
            f.write('</tr>\n')

        f.write('</table></div></body></html>\n')

    print(f"Test complete. Results saved to {output_html}")

if __name__ == "__main__":
    main()
