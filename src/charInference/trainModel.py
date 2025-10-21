from utils.charInferenceUtils import loadModel, prepareDataset, trainModel, evaluateModel, exportModel

if __name__ == "__main__":
    
    cleanDirectories(OUTPUT_DIRS)

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
