from utils.charInferenceUtils import loadModel, prepareDataset, predictImage

if __name__ == "__main__":
        
    # Test prediction on a sample image    
    model = loadModel("best.pt", "./models/yoloCharInference/train/weights")

    # Prepare dataset
    data_yaml, data_dict = prepareDataset()
    
    #Update model classes to match dataset
    model.model.names = data_dict["names"]
    model.model.nc = data_dict["nc"]
    print(f"\nModel classes set: {model.model.names}, nc={model.model.nc}")

    print("\n---\n")    
    prediction = predictImage(model, imagePath="./assets/charImages/char_4.jpg", crop=False) 
    prediction = predictImage(model, imagePath="./assets/charImages/char_9.jpg", crop=False) 
    prediction = predictImage(model, imagePath="./assets/charImages/char_V.jpg", crop=False) 
    prediction = predictImage(model, imagePath="./assets/charImages/char_5.jpg", crop=False) 
    prediction = predictImage(model, imagePath="./assets/charImages/char_1.jpg", crop=False) 
    prediction = predictImage(model, imagePath="./assets/charImages/char_3.png", crop=False) 
   