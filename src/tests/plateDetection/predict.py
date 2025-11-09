from utils.plateDetectionUtils import cleanDirectories, loadModel, prepareDataset, predictImage

if __name__ == "__main__":
    basePath = Path(__file__).parent.resolve()

    # Clean folders before running
    cleanDirectories()
    
    # Load model
    model = loadModel("best.pt", "./models/yoloPlateDetection/train/weights")

    # Prepare dataset
    data_yaml, data_dict = prepareDataset("dataset/License-Plate-Data")
    
    #Update model classes to match dataset
    model.model.names = data_dict["names"]
    model.model.nc = data_dict["nc"]
    print(f"Model classes set: {model.model.names}, nc={model.model.nc}")

    # Predict
    predictImage(model, "./assets/testImages/test-noPlate.jpg")
    predictImage(model, "./assets/info/plate.jpeg")
    predictImage(model, "./assets/info/plate2.jpg")

