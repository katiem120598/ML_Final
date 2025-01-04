from PIL import Image
import numpy as np
from CLIP_helper import *
import os

#in this example we will project multiple words along two semantic axes

#words to project
words = ["cat", "dog", "cow", "deer", "mammal", "animal", "bird", "fish", "rabbit", "turtle", "snake", "lizard", "hamster", "parrot", "penguin", "owl", "eagle", "hawk", "fur", "feather", "fly", "run"]

#semantic axes
axis1 = "mammals"
axis2 = "birds"

#we can precompute the vectors for the two axes
axis1_vec = encode_text(axis1)
axis2_vec = encode_text(axis2)

#initialize lists to store the coordinates of the words and their colors
x_data = []
y_data = []
color_data = []

#iterate over the words
for word in words:
    #encode the word
    word_vec = encode_text(word)
    #compute the similarity of the word to the two axes using the dot product (this is the cosine similarity since encode_text already normalized the vectors)
    x = np.dot(word_vec, axis1_vec)
    y = np.dot(word_vec, axis2_vec)
    #store the coordinates and the color of the word
    x_data.append(x)
    y_data.append(y)
    color_data.append((0.0, 0.0, 0.0, 0.2))


#import matplotlib for plotting
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
ax.grid(color = grid_color)


ax.set_facecolor(bg_color)
ax.axhline(0, color=axes_color)
ax.axvline(0, color=axes_color)
plt.xlabel(axis1)
plt.ylabel(axis2)

plt.xlim(-1, 1)
plt.ylim(-1, 1)


#create a scatter plot with the words as dots
ax.scatter(x_data, y_data, color=color_data)

#annotate the words with their names
for i in range(len(words)):
    plt.annotate(words[i], (x_data[i], y_data[i]), fontsize=10, color=text_color, backgroundcolor=text_bg_color)

plt.show()
