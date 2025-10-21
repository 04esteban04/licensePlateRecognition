from utils.yoloUtils import loadModel, prepareDataset, trainModel

if __name__ == "__main__":
    
    # Load yolo default model
    model = loadModel()

    # Prepare dataset
    dataYaml, dataDict = prepareDataset("dataset/LicensePlateData")
    
    #Update model classes to match dataset
    model.model.names = data_dict["names"]
    model.model.nc = data_dict["nc"]
    print(f"\nModel classes set: {model.model.names}, nc={model.model.nc}  \n")
    
    print(f"Training model with classes: {model.model.names}\n")
    
    # Train model
    trainResults = trainModel(model, data=dataYaml, project="models/yoloPlateDetection", epochs=10, imgsz=640)
    
    print("Training completed!\n")

    # Evaluate
    metrics = evaluateModel(model, data=data_yaml)
    print("\n Validation metrics:", metrics)

    # Export
    exportModel(model)   