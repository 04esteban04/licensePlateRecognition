from pathlib import Path
from segmentation.seg_utils import detectCharacters
import cv2
import numpy as np

if __name__ == "__main__":

    imagePath = "outputs/crops/plate.jpeg"
    contours, resultImg, resizedImg = detectCharacters(imagePath)

    print(f"\nDetected {len(contours)} character bounding boxes. \n")
    print("Bounding boxes:", contours, "\n")

    # Display results
    cv2.imshow("Detected Characters", resultImg)
    cv2.imshow("Resized Output", resizedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()