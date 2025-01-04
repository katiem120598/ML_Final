#before starting make sure to install the following packages:

#A. install pytorch from https://pytorch.org/get-started/locally/

#B. install CLIP following instructions in https://github.com/openai/CLIP
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git

#the PIL library is used to open and manipulate images
from PIL import Image

#numpy is used to manipulate vectors
import numpy as np

#the CLIP_helper.py file contains the code to encode text and images using the CLIP model
from CLIP_helper import *

#os is used to interact with the file system (list contents of a folders, construct file paths, etc.)
import os


#in this example we will project the word "cat" along the "animal" and "plant" axes
#then we will use matplotlib to plot the word in the 2D plane defined by the two axes

word = "cat"

axis1 = "animal"
axis2 = "plant"

#encode the word and the two axes
word_vec = encode_text(word)
axis1_vec = encode_text(axis1)
axis2_vec = encode_text(axis2)

#compute the similarity of the word to the two axes using the dot product (this is the cosine similarity since encode_text already normalized the vectors)
x = np.dot(word_vec, axis1_vec)
y = np.dot(word_vec, axis2_vec)

#print the similarity of the word to the two axes
print(f"Similarity to {axis1}: {x}")
print(f"Similarity to {axis2}: {y}")

#import matplotlib for plotting
import matplotlib.pyplot as plt

#set the colors for the plot
bg_color = (0.35, 0.45, 0.6)
axes_color = (0.5, 1.0, 1.0)
grid_color = (0.2, 0.6, 0.7)
text_color = (0.8, 1.0, 1.0)
text_bg_color = (0.1, 0.55, 0.6, 0.4)
dot_color = (1.0, 0.0, 1.0)

#create the plot with the specified bavkground color of the window
plt.figure(figsize=(6,6), facecolor=bg_color)
#set the title of the plot
plt.title(f"Planar projection along {axis1} and {axis2}")

#enable the grid and set its color
plt.grid(color = grid_color)

#set the limits of the plot (zoom level on the XY plane)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

#set the labels for the axes
plt.xlabel(axis1)
plt.ylabel(axis2)

#getthe current plot axes (matplotlib supports multiple axes (kind of viewports) in the same plot but in this and following examples we will use only one)
ax = plt.gca()

#set the background color of the plot
ax.set_facecolor(bg_color)
#draw the X and Y axes as a horizontal and vertical lines
ax.axhline(0, color=axes_color)
ax.axvline(0, color=axes_color)

#draw the word at the specified coordinates
ax.text(x, y, word, fontsize=10, color=text_color, backgroundcolor=text_bg_color)
#draw a dot at the specified coordinates
ax.scatter([x], [y], color=dot_color)

#show the plot
plt.show()
