import cv2
import numpy as np
import math

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
    
    class_idx = 1
    concept = classifier.class_names[class_idx]
    text_vec = np.zeros(classifier.class_count)
    text_vec[class_idx] = 1.0

frag_x = 5
frag_y = 5

rolling_centroid_x = 0.0
rolling_centroid_y = 0.0

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

    cen_x = 0.0
    cen_y = 0.0
    sum_sim = 0.0
    for i, frag in enumerate(fragments):
        sim = frag_sims_norm[i]
        
        #write text on the fragment
        label = f'{sim:.2f}'
        cv2.putText(frag, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        x = frag_centers[i][0]
        y = frag_centers[i][1]

        #cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)

        weight = math.exp(sim*10)

        # #calculate the centroid
        cen_x += x*weight
        cen_y += y*weight
        sum_sim += weight

    cen_x /= sum_sim
    cen_y /= sum_sim

    #smooth the centroid
    rolling_centroid_x = rolling_centroid_x*0.8 + cen_x*0.2
    rolling_centroid_y = rolling_centroid_y*0.8 + cen_y*0.2

    #draw the centroid
    cv2.circle(frame, (int(rolling_centroid_x), int(rolling_centroid_y)), 10, (0, 255, 0), 2)
    

    cv2.imshow(f'frame {concept}', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
