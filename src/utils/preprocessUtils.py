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


def preprocessPlate(imagePath):
    """Main function to preprocess an image."""
    
    # Read and preprocess
    inputImg = readImage(imagePath)

    isRedPlate = detectRedColor(inputImg)
    print(f"Red color detected: {isRedPlate[0]} (Red ratio: {isRedPlate[1]:.4f})")

    grayImg = toGrayscale(inputImg)
    grayImg = reduceNoise(grayImg)
    threshImg = applyBinarization(grayImg)

    # Morphological operations
    threshImg = closeGaps(threshImg)
    threshImg = dilateThinCharacters(threshImg)

    return inputImg, threshImg, isRedPlate
