from PIL import Image
import numpy as np
from CLIP_helper import *
import os

#here we are going to plot phrases along two axes but this time measuring polarization
#we define two concepts
#the vertical Y axis will measure the difference between the projection of each phrase along the two concepts (either as word*concept2 - word*concept1 or word*(concept2 - concept1))
#the horizontal X axis will measure the average of the sum of the two projections |word*concept1| + |word*concept2|. Thi will be very small if the word is nrelated to both concepts and large if it is related to one of them or both
words = ["the funeral was fun", "everyone cried at the funeral", "the party was depressing", "we laughed", "we watched a comedy show", "there was a horrible accident", "party"]

concept1 = "happy happiness joy"
concept2 = "sad sadness grief"

concept1_vec = encode_text(concept1)
concept2_vec = encode_text(concept2)


x_data = []
y_data = []
color_data = []

for word in words:
    word_vec = encode_text(word)
    a1 = np.dot(word_vec, concept1_vec)
    a2 = np.dot(word_vec, concept2_vec)
    x = (abs(a1) + abs(a2))*0.25
    y = a2 - a1
    #y = np.dot(word_vec, axis1_vec) - np.dot(word_vec, axis2_vec)
    x_data.append(x)
    y_data.append(y)
    color_data.append((0.0, 0.0, 0.0, 0.2))

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
plt.title(f"Polarization along {concept2} - {concept1}")
ax = plt.gca()
ax.grid(color = grid_color)

ax.set_facecolor(bg_color)
ax.axhline(0, color=axes_color)
ax.axvline(0, color=axes_color)
plt.xlabel(f"|{concept2}| + |{concept1}|")
plt.ylabel(f"{concept2} - {concept1}")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

ax.scatter(x_data, y_data, color=color_data)
for i in range(len(words)):
    plt.annotate(words[i], (x_data[i], y_data[i]), fontsize=10, color=text_color, backgroundcolor=text_bg_color)

plt.show()