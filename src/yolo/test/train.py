from yolo.yolo_utils import loadModel, prepareDataset, trainModel

if __name__ == "__main__":
    
    model = loadModel()
    dataYaml, dataDict = prepareDataset("dataset/License-Plate-Data")
    model.model.names = dataDict["names"]
    model.model.nc = dataDict["nc"]

    print(f"Training model with classes: {model.model.names}")
    
    trainResults = trainModel(model, data=dataYaml, epochs=10, imgsz=640)
    
    print("Training completed!")