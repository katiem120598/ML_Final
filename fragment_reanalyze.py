from classifier_helper import *
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

def finalize_img(classifier_name, idx, letter):
    # Path setup
    CURRENT_PATH = os.getcwd()
    MODEL_FOLDER = os.path.join(CURRENT_PATH, classifier_name)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the classifier
    classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
    classifier.load()

    # Image directory path
    detail_images_folder = os.path.join(os.getcwd(), 'analysis_results')
    if not os.path.exists(detail_images_folder):
        print(f"Directory not found: {detail_images_folder}")
        return

    image_files = [f for f in os.listdir(detail_images_folder) if os.path.isfile(os.path.join(detail_images_folder, f))]

    # Processing images
    for file_name in image_files:
        if file_name.split('_')[0] == letter:
            image_path = os.path.join(detail_images_folder, file_name)
            image = Image.open(image_path).convert('RGB')
            activation, probability, maximum_index = classifier.classify(image)

            # Displaying results
            plt.imshow(image)
            plt.title(f"Class: {classifier.class_names[idx]} ({probability[idx]:.2f}%)")
            plt.text(20, 40, f"Class: {classifier.class_names[idx]} ({probability[idx]:.2f}%)", color='red')
            plt.text(20, 80, f"Class: {classifier.class_names[0]} ({probability[0]:.2f}%)", color='red')
            save_path = os.path.join('whole_results', f"{file_name.split('.')[0]}_{classifier_name}.jpg")
            plt.savefig(save_path,dpi=300)
            #plt.show()
            plt.clf()

# Example usage

countries = ['B', 'C', 'D', 'E', 'F']
for i in range(5):
    finalize_img('classifier_model',i+1,countries[i])
    finalize_img('classifier_model_bw',i+1, countries[i])
    finalize_img('classifier_model_small',i+1,countries[i])
