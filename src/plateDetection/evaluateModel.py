from utils.plateDetectionUtils import loadModel, prepareDataset, evaluateModel

if __name__ == "__main__":

    model = loadModel("best.pt", "./models/yoloPlateDetection/train/weights")
    dataYaml, _ = prepareDataset("dataset/LicensePlateData")

    print("\n Evaluating model... \n")
    metrics = evaluateModel(model, data=dataYaml)

    print("\n Validation metrics: \n", metrics)
