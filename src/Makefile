PYTHON=python

test:
	$(PYTHON) -m yolo.test.test

setup:
	$(PYTHON) -m yolo.test.setup

train:
	$(PYTHON) -m yolo.test.train

evaluate:
	$(PYTHON) -m yolo.test.evaluate

export:
	$(PYTHON) -m yolo.test.export

predict:
	$(PYTHON) -m yolo.test.predict

predict-bash:
	@if [ -z "$(image)" ]; then \
		echo "Usage: make predict image=./path/to/image.jpg"; \
	else \
		$(PYTHON) yolo/predict.py $(image); \
	fi

predict-windows:
	@if "$(image)"=="" ( \
		echo Usage: make predict image=./path/to/image.jpg \
	) else ( \
		$(PYTHON) yolo/predict.py $(image) \
	)
