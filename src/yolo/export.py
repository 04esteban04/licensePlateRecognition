from yolo_utils import loadModel, exportModel

if __name__ == "__main__":
    
    model = loadModel()
    exportModel(model)
    
    print("Model export completed!")