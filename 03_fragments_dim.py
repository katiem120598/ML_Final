import cv2
import numpy as np

#set to True to use CLIP, False to use a classifier that you have trained
USE_CLIP = True

if USE_CLIP:
    from CLIP_helper import *

    concept = 'eye'
    text_vec = encode_text(concept)
else:
    from classifier_helper import *

    MODEL_FOLDER = r"models\expr_brq_cub_pop"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
    classifier.load()
    
    def encode_image(image : np.ndarray):
        activation, probability, maximum_index = classifier.classify(image)
        return activation
    
    class_idx = 0
    concept = classifier.class_names[class_idx]
    text_vec = np.zeros(classifier.class_count)
    text_vec[class_idx] = 1.0

frag_x = 2
frag_y = 2

#if not in windows simply use cv2.VideoCapture(0)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    w = frame.shape[1]
    h = frame.shape[0]

    dx = w // frag_x
    dy = h // frag_y

    fragments = []
    frag_sims =  []
    frag_centers = []


    for y in range(frag_y):
        for x in range(frag_x):
            frag = frame[y*dy:(y+1)*dy, x*dx:(x+1)*dx]
            image_vec = encode_image(frag)
            similarity = np.dot(text_vec, image_vec)
            fragments.append(frag)
            frag_sims.append(similarity)
            frag_centers.append((x*dx + dx//2, y*dy + dy//2))
            
    min_sim = min(frag_sims)
    max_sim = max(frag_sims)

    frag_sims_norm = (frag_sims - min_sim) / (max_sim - min_sim)

    for i, frag in enumerate(fragments):
        similarity = frag_sims_norm[i]
        text = f"{similarity * 100:.2f}%"  # Convert to percentage
        cv2.putText(frame, text, (frag_centers[i][0], frag_centers[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow(f'frame {concept}', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
