from torchvision import transforms
from classifier_helper import *
import matplotlib.pyplot as plt

#_____________________________________________________________
#_____________________________________________________________
#this script is for testing the image transformations that will be used during training
#images are randomly transformed and cropped so that the model can learn to recognize 
#things without memorizing the exact position of the objects in the images
#_____________________________________________________________
#_____________________________________________________________

IMAGE = r'C:\Users\pan\Dropbox\ACADEMIC\GSD\2024_02_Fall_QA\07_classifiers\scripts\data\eval\misc\0.JPEG'

IMAGE_SIZE = 256

RANDOM_DATA_ROTATIONS_DEG = 40
RANDOM_DATA_RESIZE = (0.05, 1.0)


#this is a list of transofrms from the torchvision library
#we can add as many as we want and they will be applied in sequence
TRANSFORMS = [
    transforms.RandomRotation(RANDOM_DATA_ROTATIONS_DEG),   
    transforms.RandomGrayscale(0.5),
    #transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.8, 1.2), saturation=(0.0, 0.5), hue=(-0.5, 0.5)),
    #transforms.RandomPerspective(),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=RANDOM_DATA_RESIZE, ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(),

    # transforms.ToTensor(),
    # transforms.RandomErasing(),
    # transforms.ToPILImage(),

    
]

#load the image
image = Image.open(IMAGE).convert('RGB')

#we are going to apply the transformations to the same image multiple times
COUNT = 15

#iterate over the count and apply the transformations in order
images = []
for i in range(COUNT):
    x_image = image
    for xf in TRANSFORMS:
        x_image = xf(x_image)

    images.append(x_image)

#display the images in a COLxROW grid
COL = 5
ROW = COUNT // COL

#create a new figure with ROWxCOL subplots. Each axs[i][j] will be a subplot (a cell in the grid)
fig, axs = plt.subplots(ROW, COL, figsize=(10, 10))

#iterate over the images and display them in the subplots
for i in range(COUNT):
    #compute the row and column index using the division and module of the serial index of the current image
    #then extract the subplot at that position
    ax = axs[i // COL][i % COL]

    #display the image in the subplot
    ax.imshow(images[i])
    #remove the axis drawing
    ax.axis('off')

plt.show()