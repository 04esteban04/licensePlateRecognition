import shutil
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
#import kagglehub

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
    
    if Path(model_name).exists():
        shutil.move(model_name, dst)
        print(f"Model weights moved to: {dst}")

    return model




# Download latest version
#kaggle_path = kagglehub.dataset_download("sujaymann/car-number-plate-dataset-yolo-format")
#kaggle_path = Path(kaggle_path)

def prepare_yolo_dataset(local_path="dataset/License-Plate-Data"):
    """
    Use a local YOLO-format dataset and return its data.yaml path as string.
    """
    dataset_path = Path(local_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    # Look for data.yaml file
    yaml_files = list(dataset_path.rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")

    data_yaml = yaml_files[0]
    print(f"Dataset ready at: {data_yaml}")

    # Load and modify data.yaml
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    # Update paths to be absolute
    data["train"] = str((dataset_path / "train").resolve())
    data["val"] = str((dataset_path / "test").resolve())

    # Save updated data.yaml
    with open(data_yaml, "w") as f:
        yaml.dump(data, f)
    
    return str(data_yaml), data




def trainModel(model, data, epochs=100, imgsz=640, device=None, project="outputs", name="train", workers=2):
    """
    Train YOLO model with given parameters.
    """
   
    
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"Training on: {'GPU' if device != 'cpu' else 'CPU'}")
   

    return model.train(
        data = data,
        epochs = epochs,
        imgsz = imgsz,
        device = device,
        project = project,
        name = name,
        workers=workers
    )


def evaluateModel(model, data, project="outputs", name="val"):
    """
    Evaluate the model on validation set.
    :param data: path to your data.yaml
    """
    return model.val(data=data, project=project, name=name)


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
    #cleanDirectories()

    # Load model
    model = loadModel()
    
    # Prepare dataset
    data_yaml, data_dict  = prepare_yolo_dataset("dataset/License-Plate-Data")
 
    #Update model classes to match dataset
    model.model.names = data_dict["names"]
    model.model.nc = data_dict["nc"]
    print(f"Model classes set: {model.model.names}, nc={model.model.nc}")

    # Train model
    trainResults = trainModel(model, data=data_yaml, epochs=10, imgsz=640)
    
    # Evaluate model
    metrics = evaluateModel(model, data=data_yaml)
    print("Validation metrics:", metrics)

    # Predict on an image
    #predictImage(model, "https://ultralytics.com/images/bus.jpg")
    predictImage(model, "./test.jpg")
    predictImage(model, "./placa_coche.jpeg")

    # Export model
    exportModel(model)    
