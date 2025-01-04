from torchvision import transforms
from classifier_helper import *
import os

#_____________________________________________________________
#_____________________________________________________________
#this script is for training the classifier model. It relies on the classifier_helper.py script
#to create and train the model by loading a pretrained imagenet model from pytorch and then 
#replacing the last layer with a new layer that has the number of outputs equal to the number of classes
#_____________________________________________________________
#_____________________________________________________________
CURRENT_PATH = os.getcwd()

#the images folder should contain subfolders with the class names and the images for each class
IMAGES_FOLDER = os.path.join(CURRENT_PATH, r'training_data')

#the folder where the trained model will be saved
MODEL_FOLDER = os.path.join(CURRENT_PATH, 'classifier_model_small')

#set the number of epochs to train the model for
NUMBER_OF_EPOCHS = 25

#set the learning rate for the optimizer
LEARNING_RATE = 0.003

#set this to True if you want to continue training a model that was previously trained and saved in the MODEL_FOLDER
CONTINUE_TRAINING = True

#set the random data transformations for the training images
RANDOM_DATA_ROTATIONS_DEG = 5
RANDOM_DATA_RESIZE = (0.2, 0.3)


#_____________________________________________________________
#_____________________________________________________________
#get the device (cpu or cuda) and create the classifier object
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)

#the batch size determines how many images are processed at once during training
#this may speed up training but may also require more memory
BATCH_SIZE = 6

TRANSFORMS = [
    transforms.RandomRotation(RANDOM_DATA_ROTATIONS_DEG),
    transforms.RandomResizedCrop(classifier.image_size, scale=RANDOM_DATA_RESIZE, ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip()
]

classifier.train(NUMBER_OF_EPOCHS, IMAGES_FOLDER, CONTINUE_TRAINING, TRANSFORMS, BATCH_SIZE, learning_rate=LEARNING_RATE)

classifier2 = Classifier.loadFromFolder(classifier.model_folder, device = torch.device("cpu"))
classifier2.saveJIT()