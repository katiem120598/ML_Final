from PIL import Image
import numpy as np
from CLIP_helper import *
import os

#in this example we load a single image but we break it into small fragments and distribute them along two axes
#in this case we can see a distribution of the regions of the image along the two axes
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILE = os.path.join(THIS_FOLDER, "data/jan-steen_twelfth-night-1668.jpg")

axis1 = "baroque"
axis2 = "abstract art"

axis1_vec = encode_text(axis1)
axis2_vec = encode_text(axis2)


x_data = []
y_data = []
images = []

#number of fragments in x and y direction
nx = 10
ny = 6

#partition image into small fragments
#we first load the image
image = Image.open(IMAGE_FILE)
for j in range(ny):
    for i in range(nx):
        #calculate the coordinates of the fragment

        #bottom left corner
        x0 = i * image.width // nx 
        y0 = j * image.height // ny

        #top right corner 
        x1 = (i + 1) * image.width // nx
        y1 = (j + 1) * image.height // ny

        #crop the fragment from the original image
        fragment = image.crop((x0, y0, x1, y1))

        #append the fragment to the list of images for display later
        images.append(fragment)

        #encode the fragment to a vector
        image_vec = encode_image(fragment)

        #project the fragment along the two axes
        x = np.dot(image_vec, axis1_vec)
        y = np.dot(image_vec, axis2_vec)
        x_data.append(x)
        y_data.append(y)
        

x_min = min(x_data)
x_max = max(x_data)
y_min = min(y_data)
y_max = max(y_data)

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

#ax.grid(color = grid_color)


ax.set_facecolor(bg_color)
ax.axhline(0, color=axes_color)
ax.axvline(0, color=axes_color)
plt.xlabel(axis1)
plt.ylabel(axis2)

plt.xlim(x_min-img_sz, x_max+img_sz)
plt.ylim(y_min-img_sz, y_max+img_sz)


for i, image in enumerate(images):
    x = x_data[i]
    y = y_data[i]
    ax.imshow(image, extent=(x - img_sz, x + img_sz, y - img_sz, y + img_sz), aspect='auto', alpha = 0.9, cmap='gray')
    

plt.show()
