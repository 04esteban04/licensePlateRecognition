import sys
from pathlib import Path
from yolo_utils import loadModel, prepareDataset, predictImage

if __name__ == "__main__":
    basePath = Path(__file__).parent.resolve()

    # Load model
    model = loadModel("train/weights/best.pt")

    # Prepare dataset
    data_yaml, data_dict = prepareDataset("dataset/License-Plate-Data")
    
    #Update model classes to match dataset
    model.model.names = data_dict["names"]
    model.model.nc = data_dict["nc"]
    print(f"Model classes set: {model.model.names}, nc={model.model.nc}")

    # Predict
    predictImage(model, str(basePath / "test" / "bus.jpg"))
    predictImage(model, str(basePath / "test" / "test.jpg"))
    predictImage(model, str(basePath / "test" / "placa.jpeg"))


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
