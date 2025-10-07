import cv2
import numpy as np
from pathlib import Path


def readImage(imagePath):
    """Read image from path and return it."""
    img = cv2.imread(str(Path(imagePath).resolve()))
    
    if img is None:
        raise FileNotFoundError(f"Image not found at: {imagePath}")
    
    return img


def toGrayscale(img):
    """Convert image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def reduceNoise(grayImg):
    """Reduce noise using Gaussian blur."""
    return cv2.GaussianBlur(grayImg, (3, 3), 0)


def applyBinarization(grayImg):
    """Apply adaptive thresholding for binarization."""
    
    return cv2.adaptiveThreshold(
        grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )


def closeGaps(threshImg):
    """Apply morphological closing to fill small gaps in characters."""
    kernelClose = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(threshImg, cv2.MORPH_CLOSE, kernelClose)


def dilateThinCharacters(threshImg):
    """Dilate thin characters slightly to prevent breaks."""
    kernelDilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    return cv2.dilate(threshImg, kernelDilate, iterations=1)


def findCharacterContours(threshImg):
    """Find external contours representing character candidates."""
    contours, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filterContours(contours):
    """Filter contours based on size, aspect ratio, and area."""
    filteredContours = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Size to include small characters
        if 6 < w < 150 and 18 < h < 200:
            aspectRatio = w / float(h)
            
            if 0.08 < aspectRatio < 1.0:
                area = cv2.contourArea(cnt)
            
                if area > 20:  # Remove small noise
                    filteredContours.append((x, y, w, h))
    
    return sorted(filteredContours, key=lambda b: b[0])


def drawBoundingBoxes(img, contours):
    """Draw bounding boxes around detected characters."""
    outputImg = img.copy()
    
    for x, y, w, h in contours:
        cv2.rectangle(outputImg, (x - 1, y - 1), (x + w, y + h), (0, 255, 0), 1)
    
    return outputImg


def resizeImage(img, width=800, height=200):
    """Resize image for better visualization."""
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def saveCharacterCrops(img, contours, imagePath, outputRoot="outputs/charCrops"):
    """
    Crop each character detected in an image and save it as char_#.ext
    inside a folder specific to the image name.
    """
    # Build output folder for this image
    imgPath = Path(imagePath)
    folderName = imgPath.stem
    ext = imgPath.suffix if imgPath.suffix else ".jpg"

    saveDir = Path(outputRoot) / folderName
    saveDir.mkdir(parents=True, exist_ok=True)

    # Save each cropped character
    for i, (x, y, w, h) in enumerate(contours, start=1):

        charCrop = img[y:y+h, x:x+w]
        cropFile = saveDir / f"char_{i}{ext}"
        cv2.imwrite(str(cropFile), charCrop)
        print(f"Saved character: {cropFile}")

    return saveDir


def detectCharacters(imagePath):
    """Main function to process an image and detect character bounding boxes."""
    
    # Read and preprocess
    img = readImage(imagePath)
    grayImg = toGrayscale(img)
    grayImg = reduceNoise(grayImg)
    threshImg = applyBinarization(grayImg)

    # Morphological operations
    threshImg = closeGaps(threshImg)
    threshImg = dilateThinCharacters(threshImg)

    # Contour detection and filtering
    contours = findCharacterContours(threshImg)
    filteredContours = filterContours(contours)

    # Draw results
    outputImg = drawBoundingBoxes(threshImg, filteredContours)
    resizedImg = resizeImage(outputImg)

    # Crop and save each detected character
    saveDir = saveCharacterCrops(outputImg, filteredContours, imagePath)

    return filteredContours, outputImg, resizedImg
