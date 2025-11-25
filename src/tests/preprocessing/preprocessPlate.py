import utils.preprocessUtils as preprocessUtils
import cv2

if __name__ == "__main__":

    # Generate preprocessed images for testing
    imagePath = "./outputs/plateCrop/test-CLPlate.jpg"
    inputImg, threshImg, isRedPlate = preprocessUtils.preprocessPlate(imagePath)
   
    cv2.imshow("Input Image", inputImg)
    cv2.imshow("Thresholded Image", threshImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    