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

from utils.userInterfaceUtils import (
    detectPlate,
    segmentCharacters,
    inferCharacters,
    collectOutputImages,
    saveLastResult
)


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

# Calculate precisions
def calculatePrecisions(plateDetection, charPrecisions):
    """
    Safely calculate plate detection precision and mean precision across
    plate and character detections.

    Args:
        plateDetection (ultralytics.engine.results.Results): YOLO plate detection result.
        charPrecisions (list[float]): List of individual character confidences (0–1 range).

    Returns:
        float: meanPrecision
    """
    plateDetectionPrecision = None
    meanPrecision = None

    try:
        # --- Plate detection precision ---
        if (
            plateDetection
            and plateDetection[0].boxes is not None
            and getattr(plateDetection[0].boxes, "conf", None) is not None
            and len(plateDetection[0].boxes.conf) > 0
        ):
            plateDetectionPrecision = float(plateDetection[0].boxes.conf.max())
            print(f"Prediction precision (plate): {plateDetectionPrecision:.4f}")
        else:
            print("⚠️ No valid plate confidence found.")

        # --- Character precision (from list) ---
        if charPrecisions and len(charPrecisions) > 0:
            meanPrecision = ((plateDetectionPrecision or 0) + sum(charPrecisions)) / (
                len(charPrecisions) + (1 if plateDetectionPrecision is not None else 0)
            )
            meanPrecision *= 100
            print(f"\n🔹 Mean precision: {meanPrecision:.4f}")
        else:
            print("⚠️ No character detections found.")

    except Exception as e:
        print(f"⚠️ Error while extracting precisions: {e}")
        plateDetectionPrecision = None
        meanPrecision = None

    return meanPrecision

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

# Process default image
@app.route("/process/default", methods=["POST"])
def processDefault():
    try:
        # Use a fixed image path instead of uploaded file
        baseDir = os.path.dirname(os.path.abspath(__file__))
        imagePath = os.path.join(baseDir, "assets/testImages", "carplate.png")
        filename = os.path.basename(imagePath)

        if not os.path.exists(imagePath):
            return jsonify({"error": f"Fixed image not found at {imagePath}"}), 404

        # Reset and prepare environment
        plateUtils.cleanDirectories()
        resetUploadFolder()

        # Detect plate
        plateDetection, plateCropPath = detectPlate(plateDetectionModel, imagePath, PLATE_CROPS_FOLDER)
        if not plateCropPath:
            return jsonify({"error": "No license plate detected."}), 400

        # Segment characters
        contours, isRed = segmentCharacters(plateCropPath)
        if not contours:
            return jsonify({"error": "No characters detected on the license plate."}), 400

        # Infer characters
        charInferenceResults, plateNumber = inferCharacters(
            charModel, contours, filename, CHAR_CROPS_FOLDER, isRed
        )

        # Get precisions
        meanPrecision = calculatePrecisions(plateDetection, charInferenceResults)
        print(f"\n Final Mean Precision: {meanPrecision:.4f}%")

        # Collect output images
        images = collectOutputImages(filename, Path(filename).suffix)

        # Build result data
        resultData = {
            "message": "Image processed successfully!",
            "predictedPlateNumber": plateNumber,
            "images": images,
            "originalFilename": filename,
            "timestamp": datetime.now().isoformat(),
            "meanPrecision": f"{meanPrecision:.4f}"
        }

        # Save last result
        saveLastResult(app.config["OUTPUT_FOLDER"], resultData)

        return jsonify(resultData), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process image TEST: {str(e)}"}), 500

# Process uploaded image
@app.route("/process/image", methods=["POST"])
def processImage():
    try:
        file = request.files.get("file")
        if not file or not allowedFile(file.filename):
            return jsonify({"error": "Unsupported or missing file."}), 400

        # Reset and prepare environment
        plateUtils.cleanDirectories()
        resetUploadFolder()

        filename = secure_filename(file.filename)
        imagePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(imagePath)

        # Detect plate
        plateDetection, plateCropPath = detectPlate(plateDetectionModel, imagePath, PLATE_CROPS_FOLDER)
        if not plateCropPath:
            return jsonify({"error": "No license plate detected."}), 400

        # Segment characters
        contours, isRed = segmentCharacters(plateCropPath)
        if not contours:
            return jsonify({"error": "No characters detected on the license plate."}), 400

        # Infer characters
        charInferenceResults, plateNumber = inferCharacters(
            charModel, contours, filename, CHAR_CROPS_FOLDER, isRed
        )

        # Get precisions
        meanPrecision = calculatePrecisions(plateDetection, charInferenceResults)
        print(f"\n Final Mean Precision: {meanPrecision:.4f}%")

        # Collect output images
        images = collectOutputImages(filename, Path(filename).suffix)

        # Build result data
        resultData = {
            "message": "Image processed successfully!",
            "predictedPlateNumber": plateNumber,
            "images": images,
            "originalFilename": filename,
            "timestamp": datetime.now().isoformat(),
            "meanPrecision": f"{meanPrecision:.4f}"
        }

        # Save last result
        saveLastResult(app.config["OUTPUT_FOLDER"], resultData)

        return jsonify(resultData), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

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

        plate_number = data.get("predictedPlateNumber", "N/A")
        images = data.get("images", {})
        file_name = data.get("originalFilename", "Unknown")
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
        add_image("Detected Plate", images.get("detectedPlate"), is_first=True)
        add_image("Cropped Plate", images.get("croppedPlate"))
        add_image("Segmentation (with Boxes)", images.get("segmentationBoxes"))
        add_image("Segmentation (Threshold)", images.get("segmentationThreshold"))

        # === Detected Characters Section ===
        chars = images.get("charInference", [])
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
