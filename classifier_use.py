from classifier_helper import *

#_____________________________________________________________
#_____________________________________________________________
#this script demonstrates how to use the classifier to classify images
#it laods an image and then uses the classifier to get the activations and probabilities (softmax)
#_____________________________________________________________
#_____________________________________________________________

#the folder where the trained model is saved
MODEL_FOLDER = 'D:/ML/CLASS_DEMOS/classifier_models'

#the image files to classify
image_files = [
    r"D:\Dropbox\ACADEMIC\GSD\2024_02_Fall_QA\07_classifiers_autoencoders\scripts\data\train\A_Expressionism\amedeo-modigliani_chaim-soutine.jpg",
    r"D:\Dropbox\ACADEMIC\GSD\2024_02_Fall_QA\07_classifiers_autoencoders\scripts\data\train\B_Baroque\rembrandt_the-blind-tobit-1651.jpg"
]

#load the classifier model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
classifier.load()

#iterate over the image files and classify each one
for image in image_files:
    #loads the image and converts it to RGB (in case we had grayscale image it would cause a crash as the classifier expects color images)
    image = Image.open(image).convert('RGB')

    #classify the image
    activation, probability, maximum_index = classifier.classify(image)

    #print the results
    print(f"Image: {image}")
    print(f"Activation: {activation}")
    print(f"Probabilities: {probability}")
    print(f"Maximum index: {maximum_index}")
    print(f"Class: {classifier.class_names[maximum_index]} ({probability[maximum_index]:.2f}%)")