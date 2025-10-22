import cv2
import numpy as np
from pathlib import Path


def readImage(imagePath):
    """Read image from path and return it."""
    img = cv2.imread(str(Path(imagePath).resolve()))
    
    if img is None:
        raise FileNotFoundError(f"Image not found at: {imagePath}")
    
    return img


def detectRedColor(img, threshold=0.1):
    """
    Detects whether an image (such as a license plate with a white background and red text)
    contains a significant amount of red color.

    Returns True if the percentage of red pixels exceeds the given threshold.

    Parameters:
        img (numpy.ndarray): Input image in BGR format.
        threshold (float): Minimum red pixel ratio (0–1) to consider the image as containing red.
    """
    if img is None:
        raise ValueError("Image is None. Please provide a valid image.")

    # Convert BGR → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color ranges in HSV
    lowerRed1 = np.array([0, 70, 50])
    upperRed1 = np.array([10, 255, 255])
    lowerRed2 = np.array([170, 70, 50])
    upperRed2 = np.array([180, 255, 255])

    # Create red color masks
    mask1 = cv2.inRange(hsv, lowerRed1, upperRed1)
    mask2 = cv2.inRange(hsv, lowerRed2, upperRed2)
    redMask = mask1 + mask2

    # Calculate red pixel ratio
    redRatio = np.sum(redMask > 0) / redMask.size
    containsRed = redRatio > threshold

    return containsRed, redRatio


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
        cropFile = saveDir / f"char_{i}{ext}"
        cv2.imwrite(str(cropFile), squareImg)
        print(f"Saved processed character: {cropFile}")

    return saveDir


def saveImages(inputImgWithBoxes, resultImg, resizedImg, outputRoot="outputs/segmentationResults", imagePath=None):
    """Save intermediate and final images for visualization."""
    saveDir = Path(outputRoot)
    saveDir.mkdir(parents=True, exist_ok=True)

    if imagePath:
        imgPath = Path(imagePath)
        baseName = imgPath.stem
        ext = imgPath.suffix if imgPath.suffix else ".png"
    else:
        baseName = "result"
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


def detectCharacters(imagePath):
    """Main function to process an image and detect character bounding boxes."""
    
    # Read and preprocess
    img = readImage(imagePath)

    isRedPlate = detectRedColor(img)
    print(f"Red color detected: {isRedPlate[0]} (Red ratio: {isRedPlate[1]:.4f})")

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
    inputImgWithBoxes = drawBoundingBoxes(img, filteredContours)
    outputImg = drawBoundingBoxes(threshImg, filteredContours)
    resizedImg = resizeImage(outputImg)
    
    # Save intermediate and final images
    saveImages(inputImgWithBoxes, outputImg, resizedImg)

    # Crop and save each detected character
    saveDir = saveCharacterCrops(outputImg, filteredContours, imagePath)

    return filteredContours, inputImgWithBoxes, outputImg, resizedImg, isRedPlate
