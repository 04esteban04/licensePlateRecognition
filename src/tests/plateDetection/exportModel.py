from utils.plateDetectionUtils import loadModel, exportModel

if __name__ == "__main__":
    
    model = loadModel("best.pt", "./models/yoloPlateDetection/train/weights")
    exportModel(model)
    