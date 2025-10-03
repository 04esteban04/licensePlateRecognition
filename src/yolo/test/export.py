from yolo.yolo_utils import loadModel, exportModel

if __name__ == "__main__":
    
    model = loadModel("../models/yolo/train/weights/best.pt")
    exportModel(model)
    
    print("Model export completed!")