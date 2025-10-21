from utils.yoloUtils import cleanDirectories, loadModel, prepareDataset, predictImage

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
    predictImage(model, "./assets/testImages/bus.jpg")
    predictImage(model, "./assets/testImages/plate.jpeg")
    predictImage(model, "./assets/testImages/plate2.jpg")


    """ 
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    imagePath = Path(sys.argv[1]).expanduser().resolve()
    
    if not imagePath.exists():
        print(f"Error: Image not found at {imagePath}")
        sys.exit(1)

    model = loadModel()
    
    print(f"Running prediction on {imagePath}...")
    cleanDirectories()
    predictImage(model, str(imagePath))
    """
