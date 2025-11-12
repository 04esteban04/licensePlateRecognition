import io
import json
import os
import shutil
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
import logging

import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for
)
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)
from werkzeug.utils import secure_filename

import psycopg
from dotenv import load_dotenv

import utils.plateDetectionUtils as plateDetectionUtils
import utils.charInferenceUtils as charInferenceUtils
import utils.datasetUtils as datasetUtils
from utils.userInterfaceUtils import (
    detectPlate,
    preprocessPlate,
    segmentCharacters,
    inferCharacters,
    collectOutputImages,
    saveLastResult
)


# App configuration
load_dotenv()

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
PLATE_CROPS_FOLDER = 'outputs/plateCrop'
CHAR_CROPS_FOLDER = 'outputs/charCrops'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TEMPLATE_DIR = os.path.abspath("UI/templates")
STATIC_DIR = os.path.abspath("UI/static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("plateDetection.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

#------------------------------------------ #
# Plate detection model configuration
#------------------------------------------ #

plateDetectionUtils.cleanDirectories()

plateDetectionModel = plateDetectionUtils.loadModel("best.pt", "./models/yoloPlateDetection/train/weights")

data_yaml, plateDataDict = plateDetectionUtils.prepareDataset("dataset/LicensePlateData")

plateDetectionModel.model.names = plateDataDict["names"]
plateDetectionModel.model.nc = plateDataDict["nc"]
logger.info(f"Model classes set: {plateDetectionModel.model.names}, nc={plateDetectionModel.model.nc}")


#------------------------------------------ #
# Character inference model configuration
#------------------------------------------ #

charModel = charInferenceUtils.loadModel("best.pt", "./models/yoloCharInference/train/weights")

datasetUtils.generateYoloCharDataset(seed=42)
charDataYaml, charDataDict = charInferenceUtils.prepareDataset()

charModel.model.names = charDataDict["names"]
charModel.model.nc = charDataDict["nc"]
logger.info(f"Model classes set: {charModel.model.names}, nc={charModel.model.nc}")


# ------------------------------- #
# Utility functions
# ------------------------------- #

def getDB():
    """Connects to PostgreSQL using .env variables."""

    return psycopg.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        autocommit=True,
    )

def getCarData(plateNumber):
    """Fetches car data from the database."""
    
    dbMatch = {
        "matched": False,
        "carId": None,
        "alerts": None,
        "inferenceId": None,
        "detectedAt": None,
    }
    
    try:
        with getDB() as conn:
            with conn.cursor() as cur:
                
                cur.execute(
                    """
                    SELECT id, full_plate, alerts
                    FROM car_data
                    WHERE UPPER(full_plate) = UPPER(%s)
                    LIMIT 1;
                    """,
                    (plateNumber,)
                )
                row = cur.fetchone()
                carID = row[0] if row else None
                alerts = row[2] if row else None

                # Insert inference record
                cur.execute(
                    """
                    INSERT INTO inference_records (detected_plate, car_id)
                    VALUES (%s, %s)
                    RETURNING id, detected_at;
                    """,
                    (plateNumber, carID)
                )
                infoRow = cur.fetchone()

                dbMatch["matched"] = carID is not None
                dbMatch["carId"] = carID
                dbMatch["alerts"] = alerts
                dbMatch["inferenceId"] = infoRow[0] if infoRow else None
                dbMatch["detectedAt"] = (
                    infoRow[1].isoformat() if infoRow and infoRow[1] else None
                )

        logger.info(
            "DB inference record saved. matched=%s carId=%s inferenceId=%s",
            dbMatch["matched"], dbMatch["carId"], dbMatch["inferenceId"]
        )
    except Exception as dbErr:
        logger.error("DB operation failed: %s", dbErr, exc_info=True)

    return dbMatch

def checkUploadFileType(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resetUploadFolder():
    folder = app.config['UPLOAD_FOLDER']

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)

def calculatePrecisions(plateDetection, charPrecisions):
    """
    Calculate plate detection precision and mean precision across plate and character detections.

    Args:
        plateDetection (ultralytics.engine.results.Results): YOLO plate detection result.
        charPrecisions (list[float]): List of individual character confidences (0–1 range).

    Returns:
        float: meanPrecision
    """
    
    plateDetectionPrecision = None
    meanPrecision = None

    try:
        
        if (
            plateDetection
            and plateDetection[0].boxes is not None
            and getattr(plateDetection[0].boxes, "conf", None) is not None
            and len(plateDetection[0].boxes.conf) > 0
        ):
            plateDetectionPrecision = float(plateDetection[0].boxes.conf.max())
            logger.info(f"Prediction precision (plate): {plateDetectionPrecision:.4f}")
        
        else:
            logger.info("No valid plate confidence found.")

        # Get individual character precision from list
        if charPrecisions and len(charPrecisions) > 0:
            meanPrecision = ((plateDetectionPrecision or 0) + sum(charPrecisions)) / (
                len(charPrecisions) + (1 if plateDetectionPrecision is not None else 0)
            )
            meanPrecision *= 100
            logger.info(f"Mean precision: {meanPrecision:.4f}")
        else:
            logger.info("No character detections found.")

    except Exception as e:
        logger.error(f"Error while extracting precisions: {e}", exc_info=True)
        plateDetectionPrecision = None
        meanPrecision = None

    return meanPrecision

def processPlateImage(imagePath, fileName):
    """Execute the plate recognition and inference process for a given image."""
    
    try:
        
        plateDetectionUtils.cleanDirectories()

        # Detect plate
        plateDetection, plateCropPath = detectPlate(plateDetectionModel, imagePath, PLATE_CROPS_FOLDER)
        if not plateCropPath:
            return {"error": "No license plate detected."}, 400

        # Preprocess plate image
        inputImg, threshImg, isRedPlate = preprocessPlate(plateCropPath)

        # Segment characters
        contours = segmentCharacters(plateCropPath, inputImg, threshImg)
        if not contours:
            return {"error": "No characters detected on the license plate."}, 400

        # Infer characters
        charInferenceResults, plateNumber = inferCharacters(
            charModel, contours, fileName, CHAR_CROPS_FOLDER, isRedPlate
        )

        # Get precisions
        meanPrecision = calculatePrecisions(plateDetection, charInferenceResults)
        logger.info(f"Final Mean Precision: {meanPrecision:.4f}%")
        
        # Get car data from DB
        carData = getCarData(plateNumber)
        logger.info(f"Car data retrieved: {carData}")
        
        # Get output images
        images = collectOutputImages(fileName, Path(fileName).suffix)

        resultData = {
            "message": "Image processed successfully!",
            "predictedPlateNumber": plateNumber,
            "images": images,
            "originalFilename": fileName,
            "timestamp": datetime.now().isoformat(),
            "meanPrecision": f"{meanPrecision:.4f}",
            "carData": carData
        }

        saveLastResult(app.config["OUTPUT_FOLDER"], resultData)

        return jsonify(resultData), 200

    except Exception as e:
        logger.error(f"Failed to process image: {e}", exc_info=True)
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

def generatePDF(data):
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
        centeredItalicStyle = ParagraphStyle(
            name="CenteredItalic",
            parent=styles["Italic"],
            alignment=TA_CENTER
        )

        alert_style = ParagraphStyle(
            name="AlertBox",
            parent=styles["Normal"],
            backColor=colors.HexColor("#FFE5E5"),
            textColor=colors.HexColor("#B30000"),
            borderColor=colors.HexColor("#B30000"),
            borderWidth=1,
            borderPadding=8,
            leading=14,
            alignment=TA_CENTER
        )
        ok_style = ParagraphStyle(
            name="OkBox",
            parent=styles["Normal"],
            backColor=colors.HexColor("#E7F5E9"),
            textColor=colors.HexColor("#0F5132"),
            borderColor=colors.HexColor("#0F5132"),
            borderWidth=1,
            borderPadding=8,
            leading=14,
            alignment=TA_CENTER
        )

        plate_number = data.get("predictedPlateNumber", "N/A")
        images = data.get("images", {})
        file_name = data.get("originalFilename", "Unknown")
        process_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alerts = data["carData"].get("alerts")

        elements = []

        # === Title and Metadata ===
        elements.append(Paragraph("License Plate Recognition Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Generated:</b> {process_time}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Source File:</b> {file_name}", styles["Normal"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"<b>Predicted Plate Number:</b> {plate_number}", styles["Heading2"]))
        elements.append(Spacer(1, 20))

        # === Alerts box ===
        if alerts != 'None':
            elements.append(Paragraph(f"<b>ALERTS:</b> {alerts}", alert_style))
        else:
            elements.append(Paragraph("<b>ALERTS:</b> --- No alerts ---", ok_style))
        elements.append(Spacer(1, 16))

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
            centeredItalicStyle
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
        logger.error(f"Error generating single-image PDF: {e}", exc_info=True)
        return jsonify({"error": f"Error generating single-image PDF: {str(e)}"}), 500

@app.context_processor
def inject_now():
    """Inject current year into templates."""
    return {'current_year': datetime.now().year}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    """Serve files from the outputs directory."""
   
    output_folder = app.config['OUTPUT_FOLDER']
    return send_from_directory(output_folder, filename)

@app.route("/process/image", methods=["POST"])
def processImage():
    file = request.files.get("file")
    if not file or not checkUploadFileType(file.filename):
        return jsonify({"error": "Unsupported or missing file."}), 400

    resetUploadFolder()
    
    fileName = secure_filename(file.filename)
    imagePath = os.path.join(app.config["UPLOAD_FOLDER"], fileName)
    file.save(imagePath)

    result, status = processPlateImage(imagePath, fileName)
    logger.info("\n\n" + "-"*50 + "\n")
    return result, status

@app.route("/process/default", methods=["POST"])
def processDefault():
    baseDir = os.path.dirname(os.path.abspath(__file__))
    imagePath = os.path.join(baseDir, "assets/testImages", "test-default.png")
    fileName = os.path.basename(imagePath)

    if not os.path.exists(imagePath):
        return jsonify({"error": f"Default image not found at {imagePath}"}), 404

    result, status = processPlateImage(imagePath, fileName)
    return result, status

@app.route('/export-pdf', methods=['GET'])
def export_pdf():
    
    try:
    
        base_dir = os.path.dirname(os.path.abspath(__file__))
        temp_result_path = os.path.join(base_dir, "outputs", "last_result.json")

        # Generate PDF for single image if last result exists
        with open(temp_result_path, "r") as f:
            data = json.load(f)
            if data:
                return generatePDF(data)
        
        return jsonify({"error": "No processed image data available for PDF generation."}), 400

    except Exception as e:
        logger.error(f"Failed to export PDF: {e}", exc_info=True)
        return jsonify({"error": f"Failed to export PDF: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
