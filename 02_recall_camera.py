import cv2
import numpy as np
import math

#set to True to use CLIP, False to use a classifier that you have trained
USE_CLIP = True

if USE_CLIP:
    from CLIP_helper import *

    axes_concept = ['happy', 'hand', 'sad']
    axes_num = len(axes_concept)

    axes_vec = np.array([encode_text(axis) for axis in axes_concept])
else:
    from classifier_helper import *

    MODEL_FOLDER = r"models\expr_brq_cub_pop"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
    classifier.load()
    
    def encode_image(image : np.ndarray):
        activation, probability, maximum_index = classifier.classify(image)
        return activation
    
    axes_concept = classifier.class_names

    axes_num = classifier.class_count    

    axes_vec = np.zeros((axes_num, axes_num))
    for i in range(axes_num):
        axes_vec[i, i] = 1.0



best_frames = []
best_sims = []

forgetfulness = 0.9995

#if not in windows simply use cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    w = frame.shape[1]
    h = frame.shape[0]

    
    image_vec = encode_image(frame)
    similarities = np.dot(axes_vec, image_vec)

    if len(best_frames) == 0:
        best_frames = [frame.copy() for _ in range(axes_num)]
        best_sims = similarities

    else :
        for i in range(axes_num):
            best_sims[i] = best_sims[i]*forgetfulness
            if similarities[i] > best_sims[i]:
                best_sims[i] = similarities[i]
                best_frames[i] = frame.copy()
                cv2.putText(best_frames[i], f'{axes_concept[i]}: {best_sims[i]:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

    vertical = np.concatenate(best_frames, axis=0)
    
    new_v_h = h
    new_v_w = vertical.shape[1] * h // vertical.shape[0]
    vertical = cv2.resize(vertical, (new_v_w, new_v_h))

    horizontal = np.concatenate([frame, vertical], axis=1)

    cv2.imshow("frame", horizontal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
