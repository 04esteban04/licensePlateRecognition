import shutil
import os
from pathlib import Path
from ultralytics import YOLO


def cleanDirectories(dirs=["outputs"]):
    """
    Remove specified directories if they already exist.
    :param dirs: Iterable with directory names to remove.
    """
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"Removing existing directory: {path}")
            shutil.rmtree(path)


def loadModel(model_name="yolo11n.pt", save_dir="outputs/models"):
    """
    Load a pretrained YOLO model and move it into outputs/models.
    The original file will no longer exist in its old location.
    """
    model = YOLO(model_name)

    # Ensure destination directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Move the original weights into outputs/
    dst = Path(save_dir) / Path(model_name).name
    shutil.move(model_name, dst)
    print(f"Model weights moved to: {dst}")

    return model


def trainModel(model, data="coco8.yaml", epochs=100, imgsz=640, device="cpu", project="outputs", name="train"):
    """
    Train YOLO model with given parameters.
    """
   
    return model.train(
        data = data,
        epochs = epochs,
        imgsz = imgsz,
        device = device,
        project = project,
        name = name,
    )


def evaluateModel(model, project="outputs", name="val"):
    """
    Evaluate the model on validation set.
    """
    return model.val(project=project, name=name)


def predictImage(model, image_url, show=True, project="outputs", name="predict"):
    """
    Run inference on a single image and save results.
    """
    
    results = model.predict(source=image_url, project=project, name=name, save=True)
    if show:
        results[0].show()

    print(f"Predictions saved to: {Path(project) / name}")
    return results


def exportModel(model, export_format="onnx", project="outputs", name="export"):
    """
    Export the model to a given format.
    """
    export_path = model.export(format=export_format, project=project, name=name)
    print(f"Model exported to: {export_path}")
    
    return Path(export_path)


if __name__ == "__main__":
    # Clean old directories before running
    cleanDirectories()

    # Load model
    model = loadModel()
    
    # Train model
    trainResults = trainModel(model, epochs=10)

    # Evaluate model
    metrics = evaluateModel(model)
    print("Validation metrics:", metrics)

    # Predict on an image
    predictImage(model, "https://ultralytics.com/images/bus.jpg")

    # Export model
    exportModel(model)
   
