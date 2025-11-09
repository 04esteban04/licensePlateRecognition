from utils.charInferenceUtils import loadModel, prepareDataset, evaluateModel

if __name__ == "__main__":

    model = loadModel("best.pt", "./models/yoloCharInference/train/weights")
    dataYaml, _ = prepareDataset()

    print("\n Evaluating model... \n")
    metrics = evaluateModel(model, data=dataYaml)

    print("\n Validation metrics: \n", metrics)
