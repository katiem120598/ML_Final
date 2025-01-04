import cv2
import numpy as np
import os
import torch
from classifier_helper import *

def fragment_analyze(image_path, model_path, class_idx, invert = False):
    # Load the classifier
    CURRENT_PATH = os.getcwd()
    IMAGE_PATH = os.path.join(CURRENT_PATH, f'detail_images/{image_path}')
    OUTPUT_PATH = os.path.join(CURRENT_PATH, f'analysis_results/{image_path.split(".")[0]}_{model_path.split(".")[0]}_{model_path}_{invert}.jpg')
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
        exit()

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
            
            # Compute similarities for class 0 and class 3
            similarity_base = np.dot(text_vec_base, image_vec)
            similarity_compare = np.dot(text_vec_compare, image_vec)
            
            fragments.append(frag)
            frag_sims_base.append(similarity_base)
            frag_sims_compare.append(similarity_compare)

    # Normalize similarities
    min_sim_base, max_sim_base = min(frag_sims_base), max(frag_sims_base)
    min_sim_compare, max_sim_compare = min(frag_sims_compare), max(frag_sims_compare)

    frag_sims_base_norm = (frag_sims_base - min_sim_base) / (max_sim_base - min_sim_base)
    frag_sims_compare_norm = (frag_sims_compare - min_sim_compare) / (max_sim_compare- min_sim_compare)

    # Remove fragments where class 3 similarity is not greater than class 0 similarity

    # ***** CHANGE THIS PART TO GET INVERSE *****
    if invert == False:
        for i, frag in enumerate(fragments):
            if frag_sims_compare_norm[i] >= frag_sims_base_norm[i]:
                fragments[i][:] = np.zeros_like(frag)
    else:
        for i, frag in enumerate(fragments):
            if frag_sims_compare_norm[i] <= frag_sims_base_norm[i]:
                fragments[i][:] = np.zeros_like(frag)

    # Combine fragments back into the original image
    output_image = np.zeros_like(image)
    for y in range(frag_y):
        for x in range(frag_x):
            idx = y * frag_x + x
            output_image[y * dy:(y + 1) * dy, x * dx:(x + 1) * dx] = fragments[idx]

    # Resize the output image
    resize_width = 800  # Desired width
    resize_height = 600  # Desired height
    output_image_resized = cv2.resize(output_image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

    current_fragment_index = None

    # Mouse hover functionality
    def mouse_callback(event, x, y, flags, param):
        global current_fragment_index

        # Map mouse position back to fragment grid
        grid_x = x * frag_x // resize_width
        grid_y = y * frag_y // resize_height

        if 0 <= grid_x < frag_x and 0 <= grid_y < frag_y:
            fragment_index = grid_y * frag_x + grid_x
            current_fragment_index = fragment_index
    """
    # Set up the window and callback
    cv2.namedWindow(f'Processed Image: {concept_compare}')
    cv2.setMouseCallback(f'Processed Image: {concept_compare}', mouse_callback)
    """

    #while True:
    overlay = output_image_resized.copy()
    cv2.imwrite(OUTPUT_PATH, overlay)

    # Display the similarity scores if hovering over a fragment
    """
    if current_fragment_index is not None:
        similarity_base = frag_sims_base_norm[current_fragment_index] * 100  # Convert to percentage
        similarity_compare = frag_sims_compare_norm[current_fragment_index] * 100  # Convert to percentage
        display_text1 = (f"{concept_base} Match: {similarity_base:.2f}%")
        display_text2 = (f"{concept_compare} Match: {similarity_compare:.2f}%")
        cv2.putText(overlay, display_text1, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(overlay, display_text2, (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    """

        # Show the image with the overlay
        #cv2.imshow(f'Processed Image: {concept_compare}', overlay)

        # Break on 'q' key
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
#fragment_analyze('D_Korean_detail.jpg', 'classifier_model', 1, invert = True)
