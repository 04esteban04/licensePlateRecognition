import utils.plateDetectionUtils as plateUtils
import utils.charInferenceUtils as charUtils
from utils.segmentationUtils import detectCharacters
from charInference.setupDataset import generateYoloCharDataset


if __name__ == "__main__":
    
    #------------------------------- #
    # Plate Detection
    #------------------------------- #

    # Clean folders before running
    plateUtils.cleanDirectories()

    # Load model
    model = plateUtils.loadModel("best.pt", "./models/yoloPlateDetection/train/weights")
    
    # Prepare plate recognition dataset
    data_yaml, data_dict = plateUtils.prepareDataset("dataset/LicensePlateData")
    
    #Update model classes to match dataset
    model.model.names = data_dict["names"]
    model.model.nc = data_dict["nc"]
    print(f"Model classes set: {model.model.names}, nc={model.model.nc}")
    
    # Setup char inference dataset
    print("\nSetting up dataset for char inference ...")
    generateYoloCharDataset(seed=42)

    # Detect plate on input image
    print("\n---\n")
    isPlateDetected = plateUtils.predictImage(model, "./assets/testImages/carplate.png", show=False)


    # If a license plate is detected, proceed to segmentation and character inference    
    if len(isPlateDetected[0].boxes) == 0:
        
        print("\nNo license plate detected. Exiting... \n")
        exit(0)

    else:
        print("\nA license plate has been detected. Proceeding to segmentation and character inference... \n")

        #------------------------------- #
        # Character Segmentation
        #------------------------------- #
        
        # Detect characters on cropped plate image
        contours, inputImgWithBoxes, resultImg, resizedImg, isRedPlate = detectCharacters("./outputs/plateCrop/carplate.png")

        print(f"\nDetected {len(contours)} character bounding boxes.\n")
        print("Bounding boxes:", contours, "\n")
        print(f"Is plate red: {isRedPlate[0]} (ratio {isRedPlate[1]:.4f}) \n")


        #------------------------------- #
        # Character Inference
        #------------------------------- #

        # Load char inference model
        charModel = charUtils.loadModel("best.pt", "./models/yoloCharInference/train/weights")

        # Prepare dataset
        charDataYaml, charDataDict = charUtils.prepareDataset()

        #Update charModel classes to match dataset
        charModel.model.names = charDataDict["names"]
        charModel.model.nc = charDataDict["nc"]
        print(f"\nModel classes set: {charModel.model.names}, nc={charModel.model.nc}")

        print("\n---\n")

        # Run inference on each cropped character image
        labels = []
        for i in range(1, 7):
            results, label = charUtils.predictImage(charModel, imagePath=f"./outputs/charCrops/carplate/char_{i}.png", show=False)    
            labels.append(label)
        
        plateNumber = "".join(labels)

        print(f"\n Predicted License Plate Number: {plateNumber} \n")