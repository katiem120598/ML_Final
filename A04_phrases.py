from PIL import Image
import numpy as np
from CLIP_helper import *
import os

#here we will plot phrases along two axes
words = ["the funeral was fun", "everyone cried at the funeral", "the party was depressing", "we laughed", "we watched a comedy show", "there was a horrible accident", "party"]

axis1 = "happiness joy"
axis2 = "sadness grief"

axis1_vec = encode_text(axis1)
axis2_vec = encode_text(axis2)


x_data = []
y_data = []
color_data = []

for word in words:
    word_vec = encode_text(word)
    x = np.dot(word_vec, axis1_vec)
    y = np.dot(word_vec, axis2_vec)
    x_data.append(x)
    y_data.append(y)
    color_data.append((0.0, 0.0, 0.0, 0.2))

#we can calculate the min and max values for the plot so that we can zoom in on the data from the start
x_min = min(x_data)
x_max = max(x_data)
y_min = min(y_data)
y_max = max(y_data)

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

#set the limits of the plot using the min and max values we calculated earlier
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

ax.scatter(x_data, y_data, color=color_data)
for i in range(len(words)):
    plt.annotate(words[i], (x_data[i], y_data[i]), fontsize=10, color=text_color, backgroundcolor=text_bg_color)

plt.show()