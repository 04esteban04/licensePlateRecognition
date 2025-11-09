from utils.charInferenceUtils import cleanDirectories, loadModel, prepareDataset, trainModel, evaluateModel, exportModel

if __name__ == "__main__":
    
    # Clean previous directories
    cleanDirectories()

    # Load YOLO model
    model = loadModel()

    # Load dataset YAML
    dataYamlPath, data = prepareDataset()

    # Train model
    results = trainModel(model, dataYamlPath)

    # Validate model
    evaluateModel(model, dataYamlPath)

    # Export model
    exportModel(model)
    
    print("\nAll done! The YOLO model is trained and ready to use.")
