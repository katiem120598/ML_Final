from PIL import Image
import numpy as np
from CLIP_helper import *
import os

#in this example we plot both images and words in the same 2D plane defined by two axes
#we also construct axes that are defined by two words each (pos-neg)
#if the axes are not constructed like word differences, then the images will cluster away from the words


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(THIS_FOLDER, "data/minimalism")
IMAGE_FILES = os.listdir(IMAGE_FOLDER)
IMAGE_FILES = [f for f in IMAGE_FILES if f.endswith(".jpg")]
IMAGE_FILES = [os.path.join(IMAGE_FOLDER, f) for f in IMAGE_FILES]
GRAYSCALE = False

words = ["geometric", "cube", "square", "line", "curved", "cmooth", "outdoors", "indoors", "solid", "structure", "building", "landscape", "nature", "field"]

axis1_pos = "architecture"
axis1_neg = "art"
axis2_pos = "landscape"
axis2_neg = "building"

#construct the vectors for the two axes as differences of the word vectors
axis1_vec = encode_text(axis1_pos) - encode_text(axis1_neg)
axis2_vec = encode_text(axis2_pos) - encode_text(axis2_neg)


x_data = []
y_data = []
data = []

#encode the images
for img in IMAGE_FILES:
    image = Image.open(img)
    #convert to grayscale
    if GRAYSCALE:
        image = image.convert("L")
    image_vec = encode_image(image)
    x = np.dot(image_vec, axis1_vec)
    y = np.dot(image_vec, axis2_vec)
    x_data.append(x)
    y_data.append(y)

    image = image.resize((64, 64), resample=Image.LANCZOS)
    data.append(image)

#encode the words
for word in words:
    word_vec = encode_text(word)
    x = np.dot(word_vec, axis1_vec)
    y = np.dot(word_vec, axis2_vec)
    x_data.append(x)
    y_data.append(y)

    data.append(word)

x_min = min(x_data)
x_max = max(x_data)
y_min = min(y_data)
y_max = max(y_data)

img_sz = (x_max - x_min) * 0.02

import matplotlib.pyplot as plt

bg_color = (0.35, 0.45, 0.6)
axes_color = (0.5, 1.0, 1.0)
grid_color = (0.2, 0.6, 0.7)
text_color = (0.8, 1.0, 1.0)
text_bg_color = (0.1, 0.55, 0.6, 0.4)
dot_color = (1.0, 0.0, 1.0)

plt.figure(figsize=(6,6), facecolor=bg_color)
plt.title(f"Planar projection along {axis1_pos} - {axis1_neg} and {axis2_pos} - {axis2_neg}")
ax = plt.gca()

ax.set_facecolor(bg_color)
ax.axhline(0, color=axes_color)
ax.axvline(0, color=axes_color)
plt.xlabel(f"{axis1_pos} - {axis1_neg}")
plt.ylabel(f"{axis2_pos} - {axis2_neg}")

plt.xlim(x_min-img_sz, x_max+img_sz)
plt.ylim(y_min-img_sz, y_max+img_sz)

for i, d in enumerate(data):
    x = x_data[i]
    y = y_data[i]
    
    #if d in words:
    if isinstance(d, str):
        ax.text(x, y, d, fontsize=10, color=text_color, backgroundcolor=text_bg_color)
    else:   
        ax.imshow(d, extent=(x - img_sz, x + img_sz, y - img_sz, y + img_sz), aspect='auto', alpha = 0.9, cmap='gray')
    

plt.show()
