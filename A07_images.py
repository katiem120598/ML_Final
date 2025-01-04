from PIL import Image
import numpy as np
from CLIP_helper import *
import os

#in this script we will project images along two axes

#get the folder of this script so we can load images from the data/images subfolder
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
#set the image folder to the subfolder data/images
IMAGE_FOLDER = os.path.join(THIS_FOLDER, "data/images")
#list all files in the image folder. This basically returns a list of strings with the file names
IMAGE_FILES = os.listdir(IMAGE_FOLDER)
#select only the files with jpg extension
IMAGE_FILES = [f for f in IMAGE_FILES if f.endswith(".jpg")]
#add the full path to the file names (This gives absolute paths like C:/.../data/images/image1.jpg)
IMAGE_FILES = [os.path.join(IMAGE_FOLDER, f) for f in IMAGE_FILES]

#We will sue this flag to turn images to grayscale so that we can see if the semantic mapping changes when we ignore color
GRAYSCALE = False

axis1 = "modernism"
axis2 = "baroque"

axis1_vec = encode_text(axis1)
axis2_vec = encode_text(axis2)


x_data = []
y_data = []

#loop over all image files
for img in IMAGE_FILES:
    #we use PIL to open the image
    image = Image.open(img)

    #convert to grayscale if the flag is set
    if GRAYSCALE:
        image = image.convert("L")

    #encode the image to a vector
    image_vec = encode_image(image)
    
    x = np.dot(image_vec, axis1_vec)
    y = np.dot(image_vec, axis2_vec)
    x_data.append(x)
    y_data.append(y)

x_min = min(x_data)
x_max = max(x_data)
y_min = min(y_data)
y_max = max(y_data)

#calculate an estimate size of the image thumbnails based on the range of the data
img_sz = (x_max - x_min) * 0.04

import matplotlib.pyplot as plt

bg_color = (0.35, 0.45, 0.6)
axes_color = (0.5, 1.0, 1.0)
grid_color = (0.2, 0.6, 0.7)
text_color = (0.8, 1.0, 1.0)
text_bg_color = (0.1, 0.55, 0.6, 0.4)
dot_color = (1.0, 0.0, 1.0)

plt.figure(figsize=(6,6), facecolor=bg_color)
plt.title(f"Planar projection along {axis1} and {axis2}")
ax = plt.gca()


ax.set_facecolor(bg_color)
ax.axhline(0, color=axes_color)
ax.axvline(0, color=axes_color)
plt.xlabel(axis1)
plt.ylabel(axis2)

plt.xlim(x_min-img_sz, x_max+img_sz)
plt.ylim(y_min-img_sz, y_max+img_sz)

#loop through the image fiels to create and display a thumbnail of each image
for i, img_file in enumerate(IMAGE_FILES):
    #load the image
    image = Image.open(img_file)

    #convert to grayscale if the flag is set
    if GRAYSCALE:
        image = image.convert("L")

    #resize to a smaller size for better performance
    image = image.resize((64, 64), resample=Image.LANCZOS)
    
    #display
    x = x_data[i]
    y = y_data[i]
    #plot the image centered at the x, y coordinates
    ax.imshow(image, extent=(x - img_sz, x + img_sz, y - img_sz, y + img_sz), aspect='auto', alpha = 0.9, cmap='gray')
    
plt.show()
