# License Plate Recognition with YOLO

This project implements a **car license plate detection system** using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection and a CNN and OCR for character recognition. 

It includes utilities to automatically prepare the dataset from Kaggle, train the model, validate it, run predictions, and crop the detected license plates from images.

<br>

## Content

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to use it](#how-to-use-it)
  - [Dataset preparation](#dataset-preparation)
  - [Model training](#model-training)
  - [Validate the trained Model](#validate-the-trained-model)
  - [Model export](#model-export)
  - [Prediction](#prediction)
  - [Test all at once](#test-all-at-once)

<br>

## Key Features

- Automatic download of the train license plate dataset in YOLO format from Kaggle.
- Custom YOLO model training and validation.
- Run predictions on local or remote images.
- Export trained models to other formats (ONNX, TorchScript, etc.).
- Automatic cropping and image saving of detected license plates.
- Organized scripts for training, validation, prediction, and utilities.

<br>

## Project Structure

```bash
src/
├── dataset/License-Plate-Data/   # Dataset downloaded from Kaggle (YOLO format)
├── models/yolo/                  # Trained and exported YOLO models (weights, ONNX, etc.)
├── outputs/                      # Prediction results and artifacts
│   ├── predict/                  # Images with YOLO detections drawn
│   └── crops/                    # Cropped license plates extracted from predictions
├── yolo/
│   ├── yolo_utils.py             # Core utilities for training, evaluation, prediction, and cropping
│   └── test/                     # Testing utilities and experiments
│       └── setup.py              # Model and training dataset set up 
│       └── train.py              # YOLO model training
│       └── evaluate.py           # Generate metrics from trained model
│       └── export.py             # Export the trained model to .onnx format
│       └── predict.py            # Inference script (runs YOLO on new images)
│       └── test.py               # Example test script for model trainig, evaluation, exporting it and running predictions
```

<br>

## Requirements

- Python >= 3.10
- [Ultralytics](https://docs.ultralytics.com) (`pip install ultralytics`)
- KaggleHub (`pip install kagglehub`)
- Torch (`pip install torch torchvision`)
- Pillow (`pip install pillow`)
- Make (Linux/macOS) or NMake/PowerShell (Windows)

<br>


# How to use it 

The following sections describe how to setup the base yolo model and dataset for license plates detection, train it, validate the results, export the model and generate some predictions with it. Also, a testing script which does all at once is described as well. 

## Dataset preparation

The dataset is automatically downloaded from _Kaggle_ when running any script that calls `prepareDataset`. It could also be done by running the command:

```bash
make setup
```

<br>

> [!NOTE]  
> It will be stored at: `dataset/License-Plate-Data/`

<br>

## Model training

Train the YOLO model for object detection on the _Kaggle_ dataset by running:

```bash
make train
```

> [!NOTE]  
> This will:
> - Download the dataset if not already available.
> - Update the `data.yaml` file with absolute paths.
> - Train the model and save the weights in `models/yolo/train/weights/best.pt`.

<br>

## Validate the trained model

It could be done by running the command:

```bash
make evaluate
```

> [!NOTE]  
> Results will be stored in `models/yolo/validation/`.

<br>

## Model Export

Export the trained model to ONNX by running:

```bash
make export
```

> [!NOTE]  
> It wil be stored at `models/yolo/train/weights/best.onnx`.

<br>

## Prediction

Run predictions on test images by running:

```bash
make predict
```

> [!NOTE]  
> This will generate:
> - Images with detections in outputs/predict/.
> - Cropped license plates in outputs/crops/.

<br>

## Test all at once

To do all the actions describe above in just one command run:

```bash
make test
```

> [!NOTE]  
> The results will be saved under:
> - `outputs/predict/<filename>` (for every prediction result)
> - `outputs/crops/<filename>` (for every result with the plate cropped)

<br>
