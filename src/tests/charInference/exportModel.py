from utils.charInferenceUtils import loadModel, exportModel

if __name__ == "__main__":
    
    model = loadModel("best.pt", "./models/yoloCharInference/train/weights")
    exportModel(model)
    