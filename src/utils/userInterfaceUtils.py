import os
import json
from datetime import datetime
from pathlib import Path

import utils.plateDetectionUtils as plateUtils
import utils.charInferenceUtils as charUtils
from utils.segmentationUtils import detectCharacters


def detectPlate(model, imagePath, outputCropFolder):
    """Detects a license plate from the given image and saves the cropped plate."""
    
    prediction = plateUtils.predictImage(model, imagePath, show=False)
    
    if len(prediction[0].boxes) == 0:
        return prediction, None
    
    croppedPath = os.path.join(outputCropFolder, os.path.basename(imagePath))
    
    return prediction, croppedPath


def segmentCharacters(platePath):
    """Detects character contours from a cropped plate."""
    
    contours, _, _, _, isRed = detectCharacters(platePath)
    return contours, isRed


def inferCharacters(model, contours, filename, charCropsFolder, isRedPlate):
    """Runs inference for each detected character."""
    
    charInferenceResults = []
    labels = []
    fileStem = Path(filename).stem
    fileExt = Path(filename).suffix

    if isRedPlate[0]:
        startIdx = max(1, len(contours) - 5)
        endIdx = len(contours) + 1
        labels.extend(["C", "L"])
    
    elif (len(contours) > 6):
        startIdx = len(contours) - 5
        endIdx = len(contours) + 1

    else:
        startIdx = 1
        endIdx = len(contours) + 1

    for i in range(startIdx, endIdx):
        charPath = Path(charCropsFolder) / fileStem / f"char_{i} ({fileStem}){fileExt}"
        result, label = charUtils.predictImage(model, imagePath=str(charPath), show=False)
        
        print('result: ', result)
        print('result value: ', result[0].boxes[0].conf[0])

        charInferenceResults.append(result[0].boxes[0].conf[0].item())
        labels.append(label)

    charUtils.renameInferenceOutputs("outputs/charInference", fileExt)

    return charInferenceResults, "".join(labels)


def collectOutputImages(file, fileExt):
    """Collects all relevant output image paths for the response."""
    
    filename = Path(file).stem

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
    return images


def saveLastResult(outputFolder, resultData):
    """Saves the latest result JSON for later PDF export."""
    
    tempResultPath = os.path.join(outputFolder, "last_result.json")
    os.makedirs(os.path.dirname(tempResultPath), exist_ok=True)
    
    with open(tempResultPath, "w", encoding="utf-8") as f:
        json.dump(resultData, f, indent=4, ensure_ascii=False)
    
    return tempResultPath
