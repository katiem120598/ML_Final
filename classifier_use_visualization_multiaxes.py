from classifier_helper import *
import numpy as np
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CURRENT_PATH = os.getcwd()
MODEL_FOLDER = os.path.join(CURRENT_PATH, 'classifier_model')
IMAGES_FOLDER = os.path.join(CURRENT_PATH,'testing_data/japan_testing')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
classifier.load()

def is_image_file(file:str) -> bool:
    return file.lower().endswith(".jpg") or file.lower().endswith(".png") or file.lower().endswith(".jpeg")

image_files = [os.path.join(IMAGES_FOLDER, f) for f in os.listdir(IMAGES_FOLDER) if is_image_file(f)]

class_count = len(classifier.class_names)
class_names = classifier.class_names

axis_vectors = []
for i in range(class_count):
    angle = i * 2 * math.pi / class_count
    x = math.cos(angle)
    y = math.sin(angle)
    axis_vectors.append(np.array([x, y], dtype=np.float32))


x_data = []
y_data = []

for image_file in image_files:
    image = Image.open(image_file).convert('RGB')

    activation, probability, maximum_index = classifier.classify(image)

    sum_x = 0
    sum_y = 0

    for i in range(class_count):
        sum_x += activation[i] * axis_vectors[i][0]
        sum_y += activation[i] * axis_vectors[i][1]

    x_data.append(sum_x)
    y_data.append(sum_y)

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
plt.title(f"Planar projection")
ax = plt.gca()


ax.set_facecolor(bg_color)

#add a labeled line for each axis direction
for i in range(class_count):
    x = axis_vectors[i][0]*3.0
    y = axis_vectors[i][1]*3.0
    plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, fc=axes_color, ec=axes_color)
    plt.text(x, y, class_names[i], color=text_color, backgroundcolor=text_bg_color, ha='center', va='center')


plt.xlim(x_min-img_sz, x_max+img_sz)
plt.ylim(y_min-img_sz, y_max+img_sz)

#loop through the image fiels to create and display a thumbnail of each image
for i, img_file in enumerate(image_files):
    #load the image
    image = Image.open(img_file)

    #resize to a smaller size for better performance
    image = image.resize((128, 128), resample=Image.LANCZOS)

    #add a label for each image with the file name without extensions and make it small
    plt.text(x_data[i], y_data[i]+img_sz, os.path.basename(img_file).split('.')[0], color=text_color, backgroundcolor=text_bg_color, fontsize=8, ha='center', va='center')
    
    #display
    x = x_data[i]
    y = y_data[i]
    #plot the image centered at the x, y coordinates
    ax.imshow(image, extent=(x - img_sz, x + img_sz, y - img_sz, y + img_sz), aspect='auto', alpha = 0.9, cmap='gray')
    
plt.show()
