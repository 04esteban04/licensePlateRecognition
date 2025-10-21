from utils.plateDetectionUtils import loadModel, prepareDataset, trainModel, evaluateModel, exportModel

if __name__ == "__main__":
    
    # Load yolo default model
    model = loadModel(saveDir="./models/yoloPlateDetection")

    # Prepare dataset
    dataYaml, dataDict = prepareDataset("dataset/LicensePlateData")
    
    #Update model classes to match dataset
    model.model.names = dataDict["names"]
    model.model.nc = dataDict["nc"]
    print(f"\nModel classes set: {model.model.names}, nc={model.model.nc}  \n")
    
    print(f"Training model with classes: {model.model.names}\n")
    
    # Train model
    trainResults = trainModel(model, data=dataYaml, project="models/yoloPlateDetection", epochs=10, imgsz=640)
    
    print("Training completed!\n")

    # Evaluate
    metrics = evaluateModel(model, data=dataYaml, project="models/yoloPlateDetection")
    print("\n Validation metrics:", metrics)

    # Export
    exportModel(model)   