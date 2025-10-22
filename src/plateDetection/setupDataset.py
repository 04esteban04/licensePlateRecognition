from utils.plateDetectionUtils import loadModel, prepareDataset

if __name__ == "__main__":
    
    print("\nSetting up model and dataset...")
    
    loadModel(saveDir="models/yoloPlateDetection")
    prepareDataset("dataset/LicensePlateData")
    
    print("\nSetup completed!")
