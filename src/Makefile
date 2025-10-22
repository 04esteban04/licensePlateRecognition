PYTHON=python

# ------------------------------- #
# Plate Detection Model Targets
# ------------------------------- #
setupPlateDetectionDataset:
	$(PYTHON) -m plateDetection.setupDataset

trainPlateDetectionModel:
	$(PYTHON) -m plateDetection.trainModel

testPlateDetectionModel:
	$(PYTHON) -m plateDetection.testModel

evaluatePlateDetectionModel:
	$(PYTHON) -m plateDetection.evaluateModel

exportPlateDetectionModel:
	$(PYTHON) -m plateDetection.exportModel


# ------------------------------- #
# Character Segmentation Targets
# ------------------------------- #
segmentation:
	$(PYTHON) -m segmentation.detectChar $(IMG)


# ------------------------------- #
# Character Inference Model Targets
# ------------------------------- #
setupCharInferenceDataset:
	$(PYTHON) -m charInference.setupDataset

trainCharInferenceModel:
	$(PYTHON) -m charInference.trainModel

testCharInferenceModel:
	$(PYTHON) -m charInference.testModel

evaluateCharInferenceModel:
	$(PYTHON) -m charInference.evaluateModel

exportCharInferenceModel:
	$(PYTHON) -m charInference.exportModel
