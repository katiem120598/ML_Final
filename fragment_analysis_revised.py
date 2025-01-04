import cv2
import numpy as np
import os
import torch
from classifier_helper import *

def fragment_analyze(image_path, model_path, class_idx, invert = False):
    # Load the classifier
    CURRENT_PATH = os.getcwd()
    IMAGE_PATH = os.path.join(CURRENT_PATH, f'detail_images/{image_path}')
    OUTPUT_PATH = os.path.join(CURRENT_PATH, f'analysis_results/{image_path.split(".")[0]}_{model_path.split(".")[0]}_analysis_in_{invert}.jpg')
    MODEL_FOLDER = os.path.join(CURRENT_PATH, f'{model_path}')
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
    classifier.load()

    def encode_image(image: np.ndarray):
        activation, probability, maximum_index = classifier.classify(image)
        return activation

    # Define classes
    class_base_idx = class_idx
    class_compare_idx = 0
    concept_base = classifier.class_names[class_base_idx]
    concept_compare = classifier.class_names[class_compare_idx]

    text_vec_base = np.zeros(classifier.class_count)
    text_vec_compare = np.zeros(classifier.class_count)
    text_vec_base[class_base_idx] = 1.0
    text_vec_compare[class_compare_idx] = 1.0

    # Load the image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Error: Unable to load the image!")
        return

    h, w, _ = image.shape
    frag_x = 10
    frag_y = 10
    dx = w // frag_x
    dy = h // frag_y

    fragments = []
    frag_sims_base = []
    frag_sims_compare = []

    # Process fragments and calculate similarity for both classes
    for y in range(frag_y):
        for x in range(frag_x):
            frag = image[y * dy:(y + 1) * dy, x * dx:(x + 1) * dx]
            image_vec = encode_image(frag)
            
            similarity_base = np.dot(text_vec_base, image_vec)
            similarity_compare = np.dot(text_vec_compare, image_vec)
            
            fragments.append(frag)
            frag_sims_base.append(similarity_base)
            frag_sims_compare.append(similarity_compare)

    # Normalize similarities
    min_sim_base, max_sim_base = min(frag_sims_base), max(frag_sims_base)
    min_sim_compare, max_sim_compare = min(frag_sims_compare), max(frag_sims_compare)

    frag_sims_base_norm = (frag_sims_base - min_sim_base) / (max_sim_base - min_sim_base)
    frag_sims_compare_norm = (frag_sims_compare - min_sim_compare) / (max_sim_compare - min_sim_compare)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(fragments)):
        y = i // frag_x
        x = i % frag_x
        y_start = y * dy
        y_end = (y + 1) * dy
        x_start = x * dx
        x_end = (x + 1) * dx
        if invert == False:
            if frag_sims_base_norm[i] <= frag_sims_compare_norm[i]:
                mask[y_start:y_end, x_start:x_end] = 255  # Use 255 to mark areas for inpainting
        else:
            if frag_sims_base_norm[i] >= frag_sims_compare_norm[i]:
                mask[y_start:y_end, x_start:x_end] = 255

    # Apply inpainting
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Save or display the result
    cv2.imwrite(OUTPUT_PATH, inpainted_image)
    #cv2.imshow('Inpainted Image', inpainted_image)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

# Call the function with your specific parameters
#fragment_analyze('D_Korean_detail.jpg', 'classifier_model', 1, invert = True)
