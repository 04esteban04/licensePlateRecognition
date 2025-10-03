from yolo.yolo_utils import loadModel, prepareDataset, evaluateModel

if __name__ == "__main__":

    model = loadModel("../models/yolo/train/weights/best.pt")
    dataYaml, _ = prepareDataset("dataset/License-Plate-Data")

    print("Evaluating model...")

    metrics = evaluateModel(model, data=dataYaml)

    print("\n Validation metrics: \n", metrics)
