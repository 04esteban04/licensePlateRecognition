import cv2
import numpy as np
from pathlib import Path

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


def saveCharacterCrops(img, contours, imagePath, outputRoot="outputs/charCrops", targetSize=64, border=4):
    """
    Crop each character detected in an image, center it in a square black background,
    apply binarization and morphology, and save it for YOLO training.

    Parameters:
        img (ndarray): Original image (grayscale)
        contours (list of tuples): List of bounding boxes (x, y, w, h)
        imagePath (str or Path): Path to the original image
        outputRoot (str): Root folder for saved character crops
        targetSize (int): Size of the square output image (default 64x64)
    """

    # Build output folder for this image
    imgPath = Path(imagePath)
    folderName = imgPath.stem
    ext = imgPath.suffix if imgPath.suffix else ".png"
    saveDir = Path(outputRoot) / folderName
    saveDir.mkdir(parents=True, exist_ok=True)

    # Save each cropped character
    for i, (x, y, w, h) in enumerate(contours, start=1):
        charCrop = img[y:y+h, x:x+w]

        # Resize proportionally to fit target size with border
        scale = min((targetSize - 6*border)/w, (targetSize - 6*border)/h)
        newW, newH = int(w*scale), int(h*scale)
        resizedChar = cv2.resize(charCrop, (newW, newH))

        # Create square black background
        squareImg = np.zeros((targetSize, targetSize), dtype=resizedChar.dtype)

        # Center the character
        xOffset = (targetSize - newW) // 2
        yOffset = (targetSize - newH) // 2
        squareImg[yOffset:yOffset+newH, xOffset:xOffset+newW] = resizedChar

        # Binarization
        _, squareImg = cv2.threshold(squareImg, 127, 255, cv2.THRESH_BINARY)

        # Optional morphology (slight open/close)
        kernel = np.ones((2, 2), np.uint8)
        if np.random.rand() > 0.5:
            squareImg = cv2.morphologyEx(squareImg, cv2.MORPH_OPEN, kernel)
        else:
            squareImg = cv2.morphologyEx(squareImg, cv2.MORPH_CLOSE, kernel)
       
        # Save processed character
        cropFile = saveDir / f"char_{i} ({folderName}){ext}"
        cv2.imwrite(str(cropFile), squareImg)
        print(f"Saved processed character: {cropFile}")

    return saveDir


def saveImages(originalImg, inputImgWithBoxes, resultImg, resizedImg, outputRoot="outputs/segmentationResults", imagePath=None):
    """Save intermediate and final images for visualization."""
    saveDir = Path(outputRoot)
    saveDir.mkdir(parents=True, exist_ok=True)

    if imagePath:
        imgPath = Path(imagePath)
        baseName = imgPath.stem
        ext = imgPath.suffix if imgPath.suffix else ".png"
    else:
        baseName = Path(originalImg).stem + "_result"
        ext = ".png"

    inputImgPath = saveDir / f"{baseName}_inputWithBoxes{ext}"
    resultImgPath = saveDir / f"{baseName}_threshWithBoxes{ext}"
    resizedImgPath = saveDir / f"{baseName}_resized{ext}"

    cv2.imwrite(str(inputImgPath), inputImgWithBoxes)
    cv2.imwrite(str(resultImgPath), resultImg)
    cv2.imwrite(str(resizedImgPath), resizedImg)

    print(f"Saved input image with boxes: {inputImgPath}")
    print(f"Saved thresholded image with boxes: {resultImgPath}")
    print(f"Saved resized output image: {resizedImgPath}")


def detectCharacters(imagePath, inputImg, threshImg):
    """Main function to process an image and detect character bounding boxes."""

    # Contour detection and filtering
    contours = findCharacterContours(threshImg)
    filteredContours = filterContours(contours)

    # Draw results
    inputImgWithBoxes = drawBoundingBoxes(inputImg, filteredContours)
    outputImg = drawBoundingBoxes(threshImg, filteredContours)
    resizedImg = resizeImage(outputImg)
    
    # Save intermediate and final images
    saveImages(imagePath, inputImgWithBoxes, outputImg, resizedImg)

    # Crop and save each detected character
    saveDir = saveCharacterCrops(outputImg, filteredContours, imagePath)

    return filteredContours, inputImgWithBoxes, outputImg, resizedImg
