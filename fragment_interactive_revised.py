import cv2
import numpy as np
from PIL import Image
import time
import torch

# Set to True to use CLIP, False to use a classifier that you have trained
USE_CLIP = False

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

if USE_CLIP:
    from CLIP_helper import encode_text, encode_image  # Assuming encode_image is also here
    concept_list = ['A_International_Style', 'B_European', 'C_Japanese', 'D_Korean', 'E_Chinese', 'F_Indian']
    text_vectors = {concept: normalize(encode_text(concept)) for concept in concept_list}
else:
    from classifier_helper import Classifier
    MODEL_FOLDER = r"classifier_model_small"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
    classifier.load()
    encode_image = classifier.encode_image  # Assuming Classifier class has this method

# Change camera index to the one found suitable for the USB camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Adjust the index here

last_update_time = time.time()
update_interval = 3  # Update interval in seconds
display_texts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_update_time > update_interval:
        # Convert the entire frame to PIL Image for compatibility with the classifier
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_vec = encode_image(pil_image)
        image_vec = normalize(image_vec)  # Normalize the image vector

        display_texts = []
        if USE_CLIP:
            for concept, text_vec in text_vectors.items():
                # Calculate similarity for each concept in CLIP
                similarity = np.dot(text_vec, image_vec)
                display_texts.append(f"{concept}: {similarity * 100:.2f}%")
        else:
            # Cycle through each class index for the classifier
            for class_idx in range(len(classifier.class_names)):
                text_vec = np.zeros(len(classifier.class_names))
                text_vec[class_idx] = 1.0
                activation = encode_image(pil_image)
                probability = np.dot(text_vec, activation)
                concept = classifier.class_names[class_idx]
                display_texts.append(f"{concept}: {probability * 100:.2f}%")

        last_update_time = current_time

    # Display the frame and text
    y_offset = 30  # Starting y position to display text
    for text in display_texts:
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += 20

    cv2.imshow('frame', frame)  # Use a generic window name
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
