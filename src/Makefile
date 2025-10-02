PYTHON=python

setup:
	$(PYTHON) yolo/setup.py

train:
	$(PYTHON) yolo/train.py

evaluate:
	$(PYTHON) yolo/evaluate.py

predict:
	$(PYTHON) yolo/predict.py

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

export:
	$(PYTHON) yolo/export.py

test:
	$(PYTHON) -m yolo.test.test
