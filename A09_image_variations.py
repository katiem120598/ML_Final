from PIL import Image
import numpy as np
from CLIP_helper import *
import os

#in this example we create a set of variations of an image (rotations and hue shifts, but could add more) and distribute them along two axes
#this is a study of how the image semantics change as we apply simple filters and transformations

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILE = os.path.join(THIS_FOLDER, "data/portrait.jpg")

axis1 = "baroque"
axis2 = "pop art"

axis1_vec = encode_text(axis1)
axis2_vec = encode_text(axis2)


x_data = []
y_data = []
images = []

#create a series of rotated versions of the image
rotations = 15
image = Image.open(IMAGE_FILE)
for j in range(rotations):

    angle = j * 360 / rotations
    img_rotated = image.rotate(angle, fillcolor=(0, 0, 0))

    images.append(img_rotated)

    image_vec = encode_image(img_rotated)

    x = np.dot(image_vec, axis1_vec)
    y = np.dot(image_vec, axis2_vec)
    x_data.append(x)
    y_data.append(y)
        
#create a series of hue shifted versions of the image
hue_shifts = 15
for j in range(hue_shifts):
    shift = int(j * 360 / hue_shifts)

    image_hsv = image.convert("HSV")
    h, s, v = image_hsv.split()
    h = h.point(lambda val: (val + shift) % 256)
    
    image_hsv = Image.merge("HSV", (h, s, v))
    image_rgb = image_hsv.convert("RGB")

    images.append(image_rgb)

    image_vec = encode_image(image_rgb)

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
