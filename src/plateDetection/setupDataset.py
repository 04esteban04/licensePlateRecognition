from utils.yoloUtils import loadModel, prepareDataset

if __name__ == "__main__":
    
    print("Setting up model and dataset...")
    
    loadModel()
    prepareDataset("dataset/LicensePlateData")
    
    print("Setup completed!")
