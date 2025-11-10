PYTHON=python
.PHONY: main preprocess segmentation \
        setupPlateDetectionDataset trainPlateDetectionModel testPlateDetectionModel \
        evaluatePlateDetectionModel exportPlateDetectionModel \
        setupCharInferenceDataset trainCharInferenceModel testCharInferenceModel \
        evaluateCharInferenceModel exportCharInferenceModel


# ------------------------------- #
# Main Target
# ------------------------------- #
main:
	$(PYTHON) -m tests.main


# ------------------------------- #
# Plate Detection Model Targets
# ------------------------------- #
setupPlateDetectionDataset:
	$(PYTHON) -m tests.plateDetection.setupDataset

trainPlateDetectionModel:
	$(PYTHON) -m tests.plateDetection.trainModel

testPlateDetectionModel:
	$(PYTHON) -m tests.plateDetection.testModel

evaluatePlateDetectionModel:
	$(PYTHON) -m tests.plateDetection.evaluateModel

exportPlateDetectionModel:
	$(PYTHON) -m tests.plateDetection.exportModel


# ------------------------------- #
# Preprocessing Targets
# ------------------------------- #
preprocess:
	$(PYTHON) -m tests.plateDetection.testModel
	$(PYTHON) -m tests.preprocessing.preprocessPlate


# ------------------------------- #
# Character Segmentation Targets
# ------------------------------- #
segmentation: preprocess
	$(PYTHON) -m tests.segmentation.detectChar


# ------------------------------- #
# Character Inference Model Targets
# ------------------------------- #
setupCharInferenceDataset:
	$(PYTHON) -m utils.datasetUtils

trainCharInferenceModel:
	$(PYTHON) -m tests.charInference.trainModel

testCharInferenceModel:
	$(PYTHON) -m tests.charInference.testModel

evaluateCharInferenceModel:
	$(PYTHON) -m tests.charInference.evaluateModel

exportCharInferenceModel:
	$(PYTHON) -m tests.charInference.exportModel
