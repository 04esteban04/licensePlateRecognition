from yolo_utils import loadModel, prepareDataset

if __name__ == "__main__":
    
    print("Setting up model and dataset...")
    
    loadModel()
    prepareDataset("dataset/License-Plate-Data")
    
    print("Setup completed!")
