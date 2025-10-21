from utils.plateDetectionUtils import cleanDirectories, loadModel, prepareDataset, predictImage

if __name__ == "__main__":
    
    # Clean folders before running
    cleanDirectories()

    # Load model
    model = loadModel("best.pt", "./models/yoloPlateDetection/train/weights")
    
    # Prepare dataset
    data_yaml, data_dict = prepareDataset("dataset/LicensePlateData")
    
    #Update model classes to match dataset
    model.model.names = data_dict["names"]
    model.model.nc = data_dict["nc"]
    print(f"Model classes set: {model.model.names}, nc={model.model.nc}")
    
    # Predict
    print("\n---\n")
    predictImage(model, "./assets/testImages/bus.jpg")
    predictImage(model, "./assets/testImages/plate.jpeg")
    predictImage(model, "./assets/testImages/plate2.jpg")
    predictImage(model, "./assets/testImages/cl-test.jpg")
 