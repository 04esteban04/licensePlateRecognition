from pathlib import Path
from utils.segmentationUtils import detectCharacters
import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    
    # Usage: python -m segmentation.detectChar <imagePath>

    # Argument parser 
    parser = argparse.ArgumentParser(description="Detect characters in an image")
    parser.add_argument(
        "imagePath",
        type=str,
        help="Path to the image to process"
    )

    args = parser.parse_args()
    imagePath = args.imagePath

    # Run char detection
    contours, inputImgWithBoxes, resultImg, resizedImg = detectCharacters(imagePath)

    print(f"\nDetected {len(contours)} character bounding boxes.\n")
    print("Bounding boxes:", contours, "\n")

    # Display results
    cv2.imshow("Input image with bounding boxes", inputImgWithBoxes)
    cv2.imshow("Detected characters", resultImg)
    cv2.imshow("Resized output", resizedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
