import utils.plateDetectionUtils as plateDetectionUtils
import utils.charInferenceUtils as charInferenceUtils
import utils.preprocessUtils as preprocessUtils
import utils.segmentationUtils as segmentationUtils
from .charInference.setupDataset import generateYoloCharDataset


if __name__ == "__main__":
    
    #------------------------------- #
    # Plate Detection
    #------------------------------- #

    # Clean folders before running
    plateDetectionUtils.cleanDirectories()

    # Load model
    model = plateDetectionUtils.loadModel("best.pt", "./models/yoloPlateDetection/train/weights")
    
    # Prepare plate recognition dataset
    data_yaml, data_dict = plateDetectionUtils.prepareDataset("dataset/LicensePlateData")
    
    #Update model classes to match dataset
    model.model.names = data_dict["names"]
    model.model.nc = data_dict["nc"]
    print(f"Model classes set: {model.model.names}, nc={model.model.nc}")
    
    # Setup char inference dataset
    print("\nSetting up dataset for char inference ...")
    generateYoloCharDataset(seed=42)

    # Detect plate on input image
    print("\n---\n")
    isPlateDetected = plateDetectionUtils.predictImage(model, "./assets/testImages/test-default.png", show=False)


    # If a license plate is detected, proceed to segmentation and character inference    
    if len(isPlateDetected[0].boxes) == 0:
        
        print("\nNo license plate detected. Exiting... \n")
        exit(0)

    else:
        print("\nA license plate has been detected. Proceeding to segmentation and character inference... \n")

        #------------------------------- #
        # Preprocessing
        #------------------------------- #

        imagePath = "./outputs/plateCrop/test-default.png"
        inputImg, threshImg, isRedPlate = preprocessUtils.preprocessPlate(imagePath)
        
        #------------------------------- #
        # Character Segmentation
        #------------------------------- #
        
        # Detect characters on cropped plate image
        contours, inputImgWithBoxes, resultImg, resizedImg = segmentationUtils.detectCharacters(
            imagePath,
            inputImg,
            threshImg
        )

        print(f"\nDetected {len(contours)} character bounding boxes.\n")
        print("Bounding boxes:", contours, "\n")
        print(f"Is plate red: {isRedPlate[0]} (ratio {isRedPlate[1]:.4f}) \n")


        #------------------------------- #
        # Character Inference
        #------------------------------- #

        # Load char inference model
        charModel = charInferenceUtils.loadModel("best.pt", "./models/yoloCharInference/train/weights")

        # Prepare dataset
        charDataYaml, charDataDict = charInferenceUtils.prepareDataset()

        #Update charModel classes to match dataset
        charModel.model.names = charDataDict["names"]
        charModel.model.nc = charDataDict["nc"]
        print(f"\nModel classes set: {charModel.model.names}, nc={charModel.model.nc}")

        print("\n---\n")

        # Run inference on each cropped character image
        
        labels = []

        if isRedPlate[0]:
            start_idx = max(1, len(contours) - 5)
            end_idx = len(contours) + 1
            labels.insert(0, "C")
            labels.insert(1, "L")
        else:
            start_idx = 1
            end_idx = len(contours) + 1

        for i in range(start_idx, end_idx):
            results, label = charInferenceUtils.predictImage(
                charModel, 
                imagePath=f"./outputs/charCrops/test-default/char_{i} (test-default).png", 
                show=False
            )
            labels.append(label)
        
        plateNumber = "".join(labels)

        print(f"\n Predicted License Plate Number: {plateNumber} \n")