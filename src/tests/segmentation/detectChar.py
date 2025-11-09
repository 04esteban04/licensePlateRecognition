from pathlib import Path
from utils.segmentationUtils import detectCharacters
import utils.preprocessUtils as preprocessUtils
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    
    imagePath = "./outputs/plateCrop/test-CLPlate.jpg"
    inputImg, threshImg, isRedPlate = preprocessUtils.preprocessPlate(imagePath)

    # Run char detection
    contours, inputImgWithBoxes, resultImg, resizedImg = detectCharacters(imagePath, inputImg, threshImg)

    print(f"\nDetected {len(contours)} character bounding boxes.\n")
    print("Bounding boxes:", contours, "\n")
    print(f"Is red plate: {isRedPlate[0]} with ratio {isRedPlate[1]:.4f}\n")

    # Display results
    cv2.imshow("Input image with bounding boxes", inputImgWithBoxes)
    cv2.imshow("Detected characters", resultImg)
    cv2.imshow("Resized output", resizedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
