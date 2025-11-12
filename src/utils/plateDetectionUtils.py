import shutil
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
import kagglehub
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def cleanDirectories(dirs=["outputs"]):
    """Remove specified directories if they already exist."""

    for d in dirs:
        path = Path(d)
    
        if path.exists():
            logger.info(f"Removing directory: {path}")
            shutil.rmtree(path)


def loadModel(modelName="yolo11n.pt", saveDir="models/test"):
    """Load YOLO model from local storage or download if not found."""

    Path(saveDir).mkdir(parents=True, exist_ok=True)
    modelPath = Path(saveDir) / modelName

    if modelPath.exists():
        logger.info(f"Loading local model: {modelPath}")
        return YOLO(str(modelPath))
    
    else:
        logger.info(f"Downloading model: {modelName}")
        model = YOLO(modelName)
    
        if Path(modelName).exists():
            shutil.move(modelName, modelPath)
            logger.info(f"Model saved to: {modelPath}")
    
        return model


def prepareDataset(localPath="dataset/testDataset"):
    """Prepare YOLO dataset and update data.yaml with absolute paths."""

    datasetPath = Path(localPath).expanduser().resolve()
  
    if not datasetPath.exists():
        logger.info(f"Dataset not found in {datasetPath}. Checking Kaggle cache...")

        # Try Kaggle cache/download
        kagglePath = Path(kagglehub.dataset_download("sujaymann/car-number-plate-dataset-yolo-format"))

        if kagglePath.exists():
            datasetPath.parent.mkdir(parents=True, exist_ok=True)

            # If kagglePath already matches localPath, just reuse it
            if kagglePath.resolve() != datasetPath:
                shutil.copytree(kagglePath, datasetPath)
                logger.info(f"Dataset copied from cache to: {datasetPath}")
            else:
                logger.info(f"Dataset already available in Kaggle cache: {datasetPath}")
        else:
            logger.error("Failed to retrieve dataset from Kaggle.")
            raise FileNotFoundError("Failed to retrieve dataset from Kaggle.")

    # Fix double nested folder problem
    nestedPath = datasetPath / "License-Plate-Data"
    if nestedPath.exists() and (nestedPath / "data.yaml").exists():
        logger.info("Fixing nested folder structure...")

        for item in nestedPath.iterdir():
            shutil.move(str(item), str(datasetPath))

        shutil.rmtree(nestedPath)


    yamlFiles = list(datasetPath.rglob("data.yaml"))
    if not yamlFiles:
        logger.error(f"data.yaml not found in {datasetPath}")
        raise FileNotFoundError(f"data.yaml not found in {datasetPath}")

    dataYaml = yamlFiles[0]
    logger.info(f"Dataset ready at: {dataYaml}")

    with open(dataYaml, "r") as f:
        data = yaml.safe_load(f)

    data["train"] = str((datasetPath / "train").resolve())
    data["val"] = str((datasetPath / "test").resolve())

    with open(dataYaml, "w") as f:
        yaml.dump(data, f)

    return str(dataYaml), data


def trainModel(model, data, epochs=100, imgsz=640, device=None, project="models/yolo", name="train", workers=2):
    """Train YOLO model with given parameters."""
       
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on: {'GPU' if device != 'cpu' else 'CPU'}")

    logger.info(
        f"Starting training: epochs={epochs}, imgsz={imgsz}, device={device}, "
        f"project={project}, name={name}, workers={workers}"
    )

    return model.train(
        data = data,
        epochs = epochs,
        imgsz = imgsz,
        device = device,
        project = project,
        name = name,
        workers = workers
    )


def evaluateModel(model, data, project="models/yolo", name="validation"):
    """Evaluate the model on validation set."""
    
    logger.info(f"Evaluating model: project={project}, name={name}")
    return model.val(data=data, project=project, name=name)


def exportModel(model, export_format="onnx", project="models/yolo", name="export"):
    """Export the model to a given format."""

    logger.info(f"Exporting model: format={export_format}, project={project}, name={name}")

    export_path = model.export(format=export_format, project=project, name=name)
    logger.info(f"Model exported to: {export_path}")

    return Path(export_path)


def predictImage(model, imageUrl, show=True, project="outputs", name="plateDetection", crop=True):
    """ Run inference on a single image and save results."""
    
    # Detect original file extension
    originalExt = Path(imageUrl).suffix.lower()
    logger.info(f"Running prediction on image: {imageUrl} (ext={originalExt})")
    
    # Run inference
    results = model.predict(source=imageUrl, project=project, name=name, save=True)

    if show:
        logger.debug("Showing prediction result window...")
        results[0].show()
    
    saveDir = Path(project) / name
    logger.info(f"Predictions saved to: {saveDir}")
    
    if crop:
        logger.info("Cropping detected plates...")
        cropPlates(results)
    
    # Rename the image to use the same extension as the original file
    for imgFile in saveDir.glob("*.*"):
        newName = imgFile.with_suffix(originalExt)
        if newName != imgFile:
            shutil.move(str(imgFile), str(newName))
            logger.info(f"Renamed {imgFile.name} → {newName.name}")

    logger.debug(f"type(results) = {type(results)}")
    return results


def cropPlates(results, saveDir="outputs/plateCrop"):
    """
    Crop the license plates from the results and save them as individual images.
    """
    savePath = Path(saveDir)
    savePath.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving cropped plates to: {savePath}")

    for i, r in enumerate(results):
        imgPath = Path(r.path)  # original image path with plate detections
        img = Image.open(imgPath).convert("RGB")
        ext = imgPath.suffix

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)  # [x1, y1, x2, y2]
        logger.info(f"Found {len(boxes)} plates in image: {imgPath}")
        
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cropped = img.crop((x1, y1, x2, y2))
            
            cropFileName = savePath / f"{imgPath.stem}{ext}"
            cropped.save(cropFileName)
            
            logger.info(f"Saved cropped plate: {cropFileName}")
