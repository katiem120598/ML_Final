from PIL import Image
import numpy as np
from CLIP_helper import *
import os
from typing import Set

#in this example we are going to create a scatter plot of all the words in two texts along two axes

text1 = "Architecture is the art and technique of designing and building, as distinguished from the skills associated with construction.[3] It is both the process and the product of sketching, conceiving,[4] planning, designing, and constructing buildings or other structures.[5] The term comes from Latin architectura; from Ancient Greek ἀρχιτέκτων (arkhitéktōn) 'architect'; from ἀρχι- (arkhi-) 'chief' and τέκτων (téktōn) 'creator'. Architectural works, in the material form of buildings, are often perceived as cultural symbols and as works of art. Historical civilisations are often identified with their surviving architectural achievements. The practice, which began in the prehistoric era, has been used as a way of expressing culture by civilizations on all seven continents.[7] For this reason, architecture is considered to be a form of art. Texts on architecture have been written since ancient times. The earliest surviving text on architectural theories is the 1st century AD treatise De architectura by the Roman architect Vitruvius, according to whom a good building embodies firmitas, utilitas, and venustas (durability, utility, and beauty). Centuries later, Leon Battista Alberti developed his ideas further, seeing beauty as an objective quality of buildings to be found in their proportions. In the 19th century, Louis Sullivan declared that form follows function. Function began to replace the classical utility and was understood to include not only practical but also aesthetic, psychological, and cultural dimensions. The idea of sustainable architecture was introduced in the late 20th century."

text2 = "A landscape is the visible features of an area of land, its landforms, and how they integrate with natural or human-made features, often considered in terms of their aesthetic appeal.[1] A landscape includes the physical elements of geophysically defined landforms such as mountains, hills, water bodies such as rivers, lakes, ponds and the sea, living elements of land cover including indigenous vegetation, human elements including different forms of land use, buildings, and structures, and transitory elements such as lighting and weather conditions. Combining both their physical origins and the cultural overlay of human presence, often created over millennia, landscapes reflect a living synthesis of people and place that is vital to local and national identity. The character of a landscape helps define the self-image of the people who inhabit it and a sense of place that differentiates one region from other regions. It is the dynamic backdrop to people's lives. Landscape can be as varied as farmland, a landscape park or wilderness. The Earth has a vast range of landscapes including the icy landscapes of polar regions, mountainous landscapes, vast arid desert landscapes, islands, and coastal landscapes, densely forested or wooded landscapes including past boreal forests and tropical rainforests and agricultural landscapes of temperate and tropical regions. The activity of modifying the visible features of an area of land is referred to as landscaping"

#this function will clean up the text by removing special characters and splitting it into words
#it will also return a list of unique words (Removing duplicates)
#you can improve it using regular expressions and also bby removing words with little semantic content (like the, that, this etc..)
def cleanup_text(text : str) -> Set[str]:
    #remove special characters (this can be done more elegantly using regular expressions)
    text = text.replace(".", " ")
    text = text.replace(",", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("[", " ")
    text = text.replace("]", " ")

    #split the text at spaces " "
    words = text.split(" ")
    #remove empty words
    words = [word for word in words if len(word) > 0]
    #get unique words by constructing a set from the list
    #a python set is a special type of container that only stores unique elements
    return set(words)

#clean up the two texts to extact the set of unique words in each
words1 = cleanup_text(text1)
words2 = cleanup_text(text2)

axis1 = "architecture"
axis2 = "landscape"


axis1_vec = encode_text(axis1)
axis2_vec = encode_text(axis2)

x_data = []
y_data = []
color_data = []

for word in words1:
    word_vec = encode_text(word)
    x = np.dot(word_vec, axis1_vec)
    y = np.dot(word_vec, axis2_vec)
    x_data.append(x)
    y_data.append(y)
    color_data.append((1.0, 0.7, 0.7, 0.25))

for word in words2:
    word_vec = encode_text(word)
    x = np.dot(word_vec, axis1_vec)
    y = np.dot(word_vec, axis2_vec)
    x_data.append(x)
    y_data.append(y)
    color_data.append((0.4, 1.0, 0.5, 0.25))

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

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#plt.text(x, y, word, fontsize=10, color=text_color, backgroundcolor=text_bg_color)
ax.scatter(x_data, y_data, color=color_data)


plt.show()
# Output: