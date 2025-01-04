import cv2
import numpy as np
import math

#set to True to use CLIP, False to use a classifier that you have trained
USE_CLIP = False

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
    
    class_idx = 1
    concept = classifier.class_names[class_idx]
    text_vec = np.zeros(classifier.class_count)
    text_vec[class_idx] = 1.0


frag_x = 6
frag_y = 5


#if not in windows simply use cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.resize(frame, (640, 480))

    w = frame.shape[1]
    h = frame.shape[0]

    dx = w // frag_x
    dy = h // frag_y

    fragments = []
    frag_sims =  []
    frag_centers = []
    frag_rects = []

    frame_cen_x = w // 2
    frame_cen_y = h // 2

    rect_ordering = []

    for y in range(frag_y):
        for x in range(frag_x):
            frag_rects.append((x*dx, (x+1)*dx, y*dy, (y+1)*dy))
            frag = frame[y*dy:(y+1)*dy, x*dx:(x+1)*dx]

            image_vec = encode_image(frag)
            similarity = np.dot(text_vec, image_vec)
            fragments.append(frag)
            frag_sims.append(similarity)

            cen_x = x*dx + dx//2
            cen_y = y*dy + dy//2

            cen_dist_x = cen_x - frame_cen_x
            cen_dist_y = cen_y - frame_cen_y

            cen_dist = -math.sqrt(cen_dist_x**2 + cen_dist_y**2)

            rect_ordering.append(cen_dist)
            
            frag_centers.append((cen_x, cen_y))

    min_sim = min(frag_sims)
    max_sim = max(frag_sims)

    frag_sims_norm = (frag_sims - min_sim) / (max_sim - min_sim)

    blank = np.zeros_like(frame)

    rect_indices = np.argsort(rect_ordering)
    sorted_indices = np.argsort(frag_sims_norm)

    

    for i, f_idx in enumerate(sorted_indices):
        frag = fragments[f_idx]
        sim = frag_sims_norm[f_idx]
        dst_rect = frag_rects[rect_indices[i]]

        frag[:] = (frag + (1.0-sim)*255.0).clip(0.0, 255.0).astype(np.uint8)
        
        blank[dst_rect[2]:dst_rect[3], dst_rect[0]:dst_rect[1]] = frag
    

    cv2.imshow(f'frame {concept}', blank)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
