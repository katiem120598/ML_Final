import cv2
import numpy as np
from PIL import Image
import torch
from classifier_helper import Classifier, IMAGENET_TRANSFORM

# Set device for computation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the classifier from the folder where the model is saved
model_folder = r"classifier_model"  # Change this to the actual path of your model folder
classifier = Classifier.loadFromFolder(model_folder, device)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)  # Adjust the index if 0 does not correspond to your webcam

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the entire frame to PIL Image for compatibility with the classifier
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Classify the image using the classifier
        outputs, probabilities, class_idx = classifier.classify(pil_image)

        # Prepare the display text showing class and probability
        display_text = f"Class: {classifier.class_names[class_idx]}, Probability: {probabilities[class_idx]:.2f}%"

        # Display the class and probabilities on the frame
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Webcam', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
