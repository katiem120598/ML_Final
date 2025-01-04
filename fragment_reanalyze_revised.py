from classifier_helper import *
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def finalize_img(classifier_name, idx, letter, results):
    # Path setup
    CURRENT_PATH = os.getcwd()
    MODEL_FOLDER = os.path.join(CURRENT_PATH, classifier_name)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the classifier
    classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
    classifier.load()

    # Image directory path
    detail_images_folder = os.path.join(os.getcwd(), '3d_prints')
    if not os.path.exists(detail_images_folder):
        print(f"Directory not found: {detail_images_folder}")
        return

    image_files = [f for f in os.listdir(detail_images_folder) if os.path.isfile(os.path.join(detail_images_folder, f)) and f.split('_')[0] == letter]

    # Processing images
    for file_name in image_files:
        image_path = os.path.join(detail_images_folder, file_name)
        image = Image.open(image_path).convert('RGB')
        activation, probability, maximum_index = classifier.classify(image)

        # Save probabilities to results
        results.append({
            'File Name': file_name,
            'Classifier': classifier_name,
            'Class Name': classifier.class_names[idx],
            'Probability': probability[idx],
            'Probability of Intl Style': probability[0],
            'Class ID': idx
        })

        # Optional: Save images with classification results overlaid
        plt.imshow(image)
        plt.title(f"Class: {classifier.class_names[idx]} ({probability[idx]:.2f}%)")
        plt.savefig(os.path.join('whole_results', f"{file_name.split('.')[0]}_{classifier_name}.jpg"), dpi=300)
        plt.clf()

def process_images():
    results = []
    countries = ['B', 'C', 'D', 'E', 'F']
    for i in range(5):
        for classifier in ['classifier_model', 'classifier_model_bw', 'classifier_model_small']:
            finalize_img(classifier, i+1, countries[i], results)

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    df.to_csv('classification_results3.csv', index=False)

# Run the processing function
process_images()
