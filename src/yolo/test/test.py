from yolo.yolo_utils import *

if __name__ == "__main__":
    # Clean folders before running
    cleanDirectories()

    # Load model
    model = loadModel()
    
    # Prepare dataset
    data_yaml, data_dict = prepareDataset("dataset/License-Plate-Data")
    
    #Update model classes to match dataset
    model.model.names = data_dict["names"]
    model.model.nc = data_dict["nc"]
    print(f"Model classes set: {model.model.names}, nc={model.model.nc}")

    # Train
    trainResults = trainModel(model, data=data_yaml, epochs=10, imgsz=640)
    
    # Evaluate
    metrics = evaluateModel(model, data=data_yaml)
    print("Validation metrics:", metrics)

    # Export
    exportModel(model)   

    # Predict
    basePath = Path(__file__).parent.resolve()
    predictImage(model, str(basePath / "assets" / "bus.jpg"))
    predictImage(model, str(basePath / "assets" / "test.jpg"))
    predictImage(model, str(basePath / "assets" / "plate.jpeg"))
 