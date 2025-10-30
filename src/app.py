from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, url_for
from werkzeug.utils import secure_filename
#from preprocessing.preprocessImages import processImage, processFolder, processDefaultDataset
#from nn.resnet_NN_test import testWithDefaultDataset, testWithCustomDataset

import utils.plateDetectionUtils as plateUtils
import utils.charInferenceUtils as charUtils
from utils.segmentationUtils import detectCharacters
from charInference.setupDataset import generateYoloCharDataset

from datetime import datetime
import os
import shutil
import zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
from pathlib import Path
from io import BytesIO
from reportlab.lib.units import inch
import json
from datetime import datetime
from PIL import Image as PILImage
from reportlab.platypus import Image


# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
PLATE_CROPS_FOLDER = 'outputs/plateCrop'
CHAR_CROPS_FOLDER = 'outputs/charCrops'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TEMPLATE_DIR = os.path.abspath("UI/templates")
STATIC_DIR = os.path.abspath("UI/static")

# Initialize Flask app
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


#------------------------------- #
# Plate Detection Model
#------------------------------- #

# Clean folders before running
plateUtils.cleanDirectories()

# Load model
plateDetectionModel = plateUtils.loadModel("best.pt", "./models/yoloPlateDetection/train/weights")

# Prepare plate recognition dataset
data_yaml, plateDataDict = plateUtils.prepareDataset("dataset/LicensePlateData")

#Update model classes to match dataset
plateDetectionModel.model.names = plateDataDict["names"]
plateDetectionModel.model.nc = plateDataDict["nc"]
print(f"Model classes set: {plateDetectionModel.model.names}, nc={plateDetectionModel.model.nc}")

# Setup char inference dataset
print("\nSetting up dataset for char inference ...")
generateYoloCharDataset(seed=42)


#------------------------------- #
# Character Inference Model
#------------------------------- #

# Load char inference model
charModel = charUtils.loadModel("best.pt", "./models/yoloCharInference/train/weights")

# Prepare dataset
charDataYaml, charDataDict = charUtils.prepareDataset()

#Update charModel classes to match dataset
charModel.model.names = charDataDict["names"]
charModel.model.nc = charDataDict["nc"]
print(f"\nModel classes set: {charModel.model.names}, nc={charModel.model.nc}")

print("\n---\n")


# Check file extensions
def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create upload folder if it doesn't exist
def resetUploadFolder():
    folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Context processor to inject current year into templates
@app.context_processor
def inject_now():
    return {'current_year': datetime.now().year}

# Load the main UI
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Serve output files
@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    """Serve files from the src/outputs directory."""
    output_folder = app.config['OUTPUT_FOLDER']
    return send_from_directory(output_folder, filename)

# Process default dataset
@app.route("/process/default", methods=["POST"])
def process_default():
    resetUploadFolder()
    processDefaultDataset()
    testWithDefaultDataset()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "nn", "outputModel", "test", "allPredictions_customDataset.csv")
    
    if not os.path.exists(csv_path):
        return jsonify({
            "error": "Default prediction CSV file not found."
        }), 404

    try:
        df = pd.read_csv(csv_path)
        results = df.to_dict(orient="records")
        return jsonify({
            "message": "Default dataset was processed.",
            "results": results
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Failed to read predictions: {str(e)}"
        }), 500

# Process individual image
@app.route("/process/image", methods=["POST"])
def process_image():

    try:

        file = request.files.get('file')

        if not file or not allowedFile(file.filename):
            return jsonify({"error": "Unsupported or missing file."}), 400

        # Clean folders before running
        plateUtils.cleanDirectories()
        resetUploadFolder()
        
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # Detect plate on input image
        print("\n---\n")
        isPlateDetected = plateUtils.predictImage(plateDetectionModel, path, show=False)
        
        
        if len(isPlateDetected[0].boxes) == 0:
            return jsonify({
                "error": "No license plate detected in the input image."
            }), 400

        
        #------------------------------- #
        # Character Segmentation
        #------------------------------- #
        
        # Detect characters on cropped plate image
        contours, inputImgWithBoxes, resultImg, resizedImg, isRedPlate = detectCharacters(os.path.join(PLATE_CROPS_FOLDER, filename))

        print(f"\nDetected {len(contours)} character bounding boxes.\n")
        print("Bounding boxes:", contours, "\n")
        print(f"Is plate red: {isRedPlate[0]} (ratio {isRedPlate[1]:.4f}) \n")

        if len(contours) == 0:
            return jsonify({
                "error": "No characters detected on the license plate."
            }), 400
        

        # Run inference on each cropped character image
        labels = []

        if isRedPlate[0]:
            start_idx = max(1, len(contours) - 5)
            end_idx = len(contours) + 1
            labels.insert(0, "C")
            labels.insert(1, "L")
        else:
            start_idx = 1
            end_idx = len(contours) + 1

        file_stem = Path(filename).stem
        file_ext = Path(filename).suffix

        for i in range(start_idx, end_idx):
            
            char_path = Path(CHAR_CROPS_FOLDER) / file_stem / f"char_{i}{file_ext}"

            results, label = charUtils.predictImage(
                charModel, 
                imagePath=str(char_path),
                show=False
            )

            labels.append(label)
       
        plateNumber = "".join(labels)

        print(f"\n Predicted License Plate Number: {plateNumber} \n")



        images = {
            "detected_plate": f"/outputs/plateDetection/{filename}",
            "cropped_plate": f"/outputs/plateCrop/{filename}",
            "segmentation_boxes": "/outputs/segmentationResults/result_inputWithBoxes.png",
            "segmentation_threshold": "/outputs/segmentationResults/result_threshWithBoxes.png",
        }

        # Add the character inference images
        char_inference_urls = []
        char_inference_dir = Path("outputs/charInference")

        for subdir in char_inference_dir.glob("inference*"):
            for img_path in subdir.glob(f"char_*{file_ext}"):
                rel_path = img_path.relative_to("outputs")
                char_inference_urls.append(f"/outputs/{rel_path.as_posix()}")

        images["char_inference"] = char_inference_urls

        result_data = {
            "message": "Image processed successfully!",
            "predicted_plate_number": plateNumber,
            "images": images,
            "original_filename": filename,
            "timestamp": datetime.now().isoformat()
        }

        # Save last result to a temp json file
        temp_result_path = os.path.join(app.config['OUTPUT_FOLDER'], "last_result.json")
        os.makedirs(os.path.dirname(temp_result_path), exist_ok=True)

        with open(temp_result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

        return jsonify(result_data), 200

    except Exception as e:
        return jsonify({
            "error": f"Failed to read predictions: {str(e)}"
        }), 500

# Export PDF report for last processed image
@app.route('/export-pdf', methods=['GET'])
def export_pdf():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        temp_result_path = os.path.join(base_dir, "outputs", "last_result.json")

        # Generate PDF for single image if last result exists
        with open(temp_result_path, "r") as f:
            data = json.load(f)
            if data:
                return generate_single_image_pdf(data)
        
        return jsonify({"error": "No processed image data available for PDF generation."}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to export PDF: {str(e)}"}), 500

# Generate PDF for single image
def generate_single_image_pdf(data):
    """
    Generate a PDF report for the last processed image.
    Includes predicted plate, timestamp, and source filename.
    """
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=letter,
            rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30
        )
        styles = getSampleStyleSheet()
        elements = []

        plate_number = data.get("predicted_plate_number", "N/A")
        images = data.get("images", {})
        file_name = data.get("original_filename", "Unknown")
        process_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # === Title and Metadata ===
        elements.append(Paragraph("License Plate Recognition Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Generated:</b> {process_time}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Source File:</b> {file_name}", styles["Normal"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Predicted Plate Number:</b> {plate_number}", styles["Heading2"]))
        elements.append(Spacer(1, 20))


        # === Helper function to scale images ===
        def scaled_image(path, max_w=3*inch, max_h=1.5*inch):

            with PILImage.open(path) as img:
                w, h = img.size

            ratio = min(max_w / w, max_h / h)
            
            return Image(path, width=w * ratio, height=h * ratio)

        # === Helper function to safely add images ===
        def add_image(title, rel_path, is_first=False):
           
            if rel_path:
                img_path = os.path.join(os.getcwd(), rel_path.strip("/"))
                elements.append(Paragraph(f"<b>{title}</b>", styles["Heading4"]))
                
                if os.path.exists(img_path):
                    if is_first:
                        elements.append(scaled_image(img_path, max_w=5*inch, max_h=3*inch))
                    else:
                        elements.append(scaled_image(img_path))
                
                else:
                    elements.append(Paragraph(f"⚠️ Image not found: {rel_path}", styles["Normal"]))
                
                elements.append(Spacer(1, 12))

        # === Add main images ===
        add_image("Detected Plate", images.get("detected_plate"), is_first=True)
        add_image("Cropped Plate", images.get("cropped_plate"))
        add_image("Segmentation (with Boxes)", images.get("segmentation_boxes"))
        add_image("Segmentation (Threshold)", images.get("segmentation_threshold"))

        # === Detected Characters Section ===
        chars = images.get("char_inference", [])
        if chars:
            elements.append(Paragraph("<b>Detected Characters</b>", styles["Heading3"]))
            elements.append(Spacer(1, 8))

            row = []
            for i, char_img in enumerate(chars, start=1):
                img_path = os.path.join(os.getcwd(), char_img.strip("/"))
                if os.path.exists(img_path):
                    row.append(Image(img_path, width=0.8*inch, height=0.8*inch))
                if len(row) == 10:
                    elements.append(Table([row]))
                    row = []
            if row:
                elements.append(Table([row]))
            elements.append(Spacer(1, 16))

        # === Footer ===
        elements.append(Paragraph(
            "This report was automatically generated by the License Plate Recognition System.",
            styles["Italic"]
        ))

        doc.build(elements)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"license_plate_report_{plate_number}.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({"error": f"Error generating single-image PDF: {str(e)}"}), 500


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
