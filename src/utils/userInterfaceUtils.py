import os
import json
from datetime import datetime
from pathlib import Path
import logging

import utils.plateDetectionUtils as plateDetectionUtils
import utils.preprocessUtils as preprocessUtils
import utils.segmentationUtils as segmentationUtils
import utils.charInferenceUtils as charInferenceUtils

logger = logging.getLogger(__name__)


def detectPlate(model, imagePath, outputCropFolder):
    """Detects a license plate from the given image and saves the cropped plate."""
    
    logger.info(f"Running plate detection for image: {imagePath}")
    prediction = plateDetectionUtils.predictImage(model, imagePath, show=False)
    
    if len(prediction[0].boxes) == 0:
        logger.warning("No license plate detected in the image.")
        return prediction, None
    
    croppedPath = os.path.join(outputCropFolder, os.path.basename(imagePath))
    logger.debug(f"Cropped plate path will be: {croppedPath}")
    
    return prediction, croppedPath


def preprocessPlate(platePath):
    """Preprocesses the cropped plate image."""
    
    logger.info(f"Preprocessing plate image: {platePath}")
    inputImg, threshImg, isRedPlate = preprocessUtils.preprocessPlate(platePath)
    logger.debug(f"Preprocessing completed. isRedPlate={isRedPlate}")
    return inputImg, threshImg, isRedPlate


def segmentCharacters(platePath, inputImg, threshImg):
    """Detects character contours from a cropped plate."""
    
    logger.info(f"Segmenting characters for plate: {platePath}")
    contours, _, _, _ = segmentationUtils.detectCharacters(platePath, inputImg, threshImg)
    logger.debug(f"Found {len(contours)} character contours.")
    return contours


def inferCharacters(model, contours, filename, charCropsFolder, isRedPlate):
    """Runs inference for each detected character."""
    
    logger.info(f"Running character inference for file: {filename}")
    charInferenceResults = []
    labels = []
    fileStem = Path(filename).stem
    fileExt = Path(filename).suffix

    if isRedPlate[0]:
        startIdx = max(1, len(contours) - 5)
        endIdx = len(contours) + 1
        labels.extend(["C", "L"])
        logger.debug(f"Red plate detected. Using last 6 characters: indices {startIdx} to {endIdx - 1}")
    
    elif (len(contours) > 6):
        startIdx = len(contours) - 5
        endIdx = len(contours) + 1
        logger.debug(f"More than 6 contours. Using last 6 characters: indices {startIdx} to {endIdx - 1}")

    else:
        startIdx = 1
        endIdx = len(contours) + 1
        logger.debug(f"Using all contours: indices {startIdx} to {endIdx - 1}")

    for i in range(startIdx, endIdx):
        charPath = Path(charCropsFolder) / fileStem / f"char_{i} ({fileStem}){fileExt}"
        logger.debug(f"Inferring character from image: {charPath}")

        result, label = charInferenceUtils.predictImage(model, imagePath=str(charPath), show=False)
        
        if result and result[0].boxes is not None and len(result[0].boxes) > 0:
            conf = result[0].boxes[0].conf[0]
            logger.debug(f"Inference result label='{label}', confidence={conf:.4f}")
            charInferenceResults.append(conf.item())
            labels.append(label)
        else:
            logger.warning(f"No boxes found for character image: {charPath}")

    charInferenceUtils.renameInferenceOutputs("outputs/charInference", fileExt)

    fullPlate = "".join(labels)
    logger.info(f"Final inferred plate: {fullPlate}")

    return charInferenceResults, fullPlate


def collectOutputImages(file, fileExt):
    """Collects all relevant output image paths for the response."""
    
    filename = Path(file).stem
    logger.info(f"Collecting output images for: {filename}{fileExt}")

    images = {
        "detectedPlate": f"/outputs/plateDetection/{file}",
        "croppedPlate": f"/outputs/plateCrop/{file}",
        "segmentationBoxes": f"/outputs/segmentationResults/{filename}_result_inputWithBoxes.png",
        "segmentationThreshold": f"/outputs/segmentationResults/{filename}_result_threshWithBoxes.png",
    }

    charInferenceUrls = []
    charInferenceDir = Path("outputs/charInference")
    
    for subdir in charInferenceDir.glob("inference*"):
    
        for imgPath in subdir.glob(f"char_* ({filename}){fileExt}"):
    
            relPath = imgPath.relative_to("outputs")
            charInferenceUrls.append(f"/outputs/{relPath.as_posix()}")

    images["charInference"] = charInferenceUrls

    logger.debug(f"Collected {len(charInferenceUrls)} character inference images.")
    return images


def saveLastResult(outputFolder, resultData):
    """Saves the latest result JSON for later PDF export."""
    
    tempResultPath = os.path.join(outputFolder, "last_result.json")
    os.makedirs(os.path.dirname(tempResultPath), exist_ok=True)
    
    logger.info(f"Saving last result JSON to: {tempResultPath}")
    with open(tempResultPath, "w", encoding="utf-8") as f:
        json.dump(resultData, f, indent=4, ensure_ascii=False)
    
    return tempResultPath
