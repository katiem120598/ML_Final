import cv2
import numpy as np
import math

#set to True to use CLIP, False to use a classifier that you have trained
USE_CLIP = True

if USE_CLIP:
    #import the encode_text and encode_image functions from CLIP_helper.py
    from CLIP_helper import *

    #set the concepts you want to detect
    axes_concept = ['happy', 'hand', 'sad']
    #axes_concept = ['human', 'animal', 'plant', 'inanimate', 'abstract', 'text']
    #axes_concept = ['live', 'love', 'laugh', 'death', 'hate', 'cry']
    #axes_concept = ['front', 'side', 'top', 'back', 'bottom', 'right']    
    
    #the number of concept axes
    axes_num = len(axes_concept)

    #encode the text concepts to get the vectors as an array of size axes_num x 512. So for concept i, the CLIP vector is at axes_vec[i]
    axes_vec = np.array([encode_text(axis) for axis in axes_concept])
else:
    #import the classifier helper functions from classifier_helper.py
    from classifier_helper import *

    #the folder where the model is saved
    MODEL_FOLDER = r"models\expr_brq_cub_pop"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #create a classifier object with the model in MODEL_FOLDER
    classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
    classifier.load()
    
    #define the encode_image function that takes an image and returns the activation vector from the classifier
    def encode_image(image : np.ndarray):
        activation, probability, maximum_index = classifier.classify(image)
        return activation
    
    #here the concepts are taken from the classifier model (the class names we used when training the model)
    axes_concept = classifier.class_names

    #the number of concept axes
    axes_num = classifier.class_count    

    #initialize the axes_vec to be a diagonal matrix with 1.0 on the diagonal
    #bcause we want to read one class activation at a time for example for class 1 we want to read activation at index 1
    #therefore the vector is [0,1,0,0...] so when we classify an image and get a vector like [0.1, 0.9, 0.2, 0.3, ...]
    #we can get the activation for class 1 by the dot product which will yield 0.1*0 + 0.9*1 + 0.2*0 + 0.3*0 + ... = 0.9 = activation[1]
    axes_vec = np.zeros((axes_num, axes_num))
    for i in range(axes_num):
        axes_vec[i, i] = 1.0
    
#initialize the sounds
#import the mixer from pygame (use pip install pygame if you don't have it)
from pygame import mixer
mixer.init()

SOUND_PATH_0 = "cello.wav" 
sound_0 = mixer.Sound(SOUND_PATH_0)
sound_0.play(-1)

SOUND_PATH_1 = "Mii Plaza.mp3"
sound_1 = mixer.Sound(SOUND_PATH_1)
sound_1.play(-1)   
#we will create an array of sounds. Each sound's volume will be set based on the similarity of the image to the concept axis
sounds = [sound_1, None, sound_0, None, None, None, None]

#set the colors for the axes for visualization
axes_colors = [(200, 255, 220), (200, 225, 255), (255, 200, 220), (255, 255, 255), (255, 225, 255), (255, 255, 255)]
#set the tint colors for the frame based on the similarity to the axes
tint_colors = np.array([(255, 200, 150), (255, 255, 255), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)])

#setup the direction vectors in the image space for showing wach axis. 
#we create a direction vector for each concept axis at regular angles around the unit circle
axes_dir = []

start_angle = 3*math.pi / 2
angle_step = 2 * math.pi / axes_num
for i in range(axes_num):
    angle = start_angle + i * angle_step
    axes_dir.append([math.cos(angle), math.sin(angle)])
    
#record the rolling average of the similarities to smooth the visualization
smooth_sims = np.zeros(axes_num)

#record the rolling maximum similarity ever to normalize the similarities
rolling_max_sim_ever = 0.0

#initialize the camera using opencv. This will initiate camera capture
#if not in windows simply use cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#go into an infinite loop to capture frames from the camera
while True:
    #read a frame from the camera
    ret, frame = cap.read()

    #if the frame is not read properly, break the loop
    if not ret:
        break
    
    #get the width and height of the frame
    w = frame.shape[1]
    h = frame.shape[0]

    #get the center of the frame
    cen_x = w // 2
    cen_y = h // 2

    #encode the image to get the activation vector or the clip vector depending on the USE_CLIP flag
    image_vec = encode_image(frame)

    #calculate the similarity of the image to each of the concept axes
    #np.dot is the dot will compute all the dot products between the image vector and each of the axes vectors at once
    #when CLIP is false similarities == image_vec since we made the axes_vec a diagonal matrix
    similarities = np.dot(axes_vec, image_vec)

    #smooth the similarities by taking a rolling average
    smooth_sims = 0.9 * smooth_sims + 0.1 * similarities

    #normalize the similarities to be between 0 and 1
    min_sim = np.min(smooth_sims)
    max_sim = np.max(smooth_sims)

    rolling_max_sim_ever = rolling_max_sim_ever*0.95 + max_sim*0.05
    
    similarities_norm = (smooth_sims - min_sim) / (rolling_max_sim_ever - min_sim)

    #compute a tint color as a weighted average of the tint colors based on the similarities
    tint_color = np.zeros(3)
    for i in range(axes_num):
        sim = similarities_norm[i]
        tint_color += tint_colors[i] * sim

    tint_color/= np.sum(similarities_norm)

    #apply the tint color to the frame
    tinted_frame = frame * tint_color / 255.0
    frame = tinted_frame.astype(np.uint8)

    #x and y components of the centorid in the multiaxial projection of the current frame
    centroid_x = 0.0
    centroid_y = 0.0

    #loop to draw the axes and compute the centroid
    for i, axis_name in enumerate(axes_concept):
        #get the similarity of the image to the axis
        sim = similarities_norm[i]
        #create a label for the axis
        label = f"{axis_name}: {sim:.2f}"
        #get the direction vector of the axis. This is a 2d vector in the image space
        dir = axes_dir[i]
        #get the color of the axis
        color = axes_colors[i]
        #scaling factor for the axes since our similarities are between 0 and 1 and our image frame is in pixels you can think 
        #of this number as the length in pixels of the longest axis (highest activation)
        length = 200

        #compute the end point of the axis by adding the direction to the center of the frame
        x2 = cen_x + int(length * dir[0] * sim)
        y2 = cen_y + int(length * dir[1] * sim)

        #set the volume of the sound based on the similarity
        if sounds[i] is not None:
            sounds[i].set_volume(sim)
            # if sim > 0.5:
            #     sounds[i].set_volume(sim)
            #     sounds[i].play(-1)
            # else:
            #     sounds[i].stop()

        #add the end point to the centroid
        centroid_x += x2
        centroid_y += y2

        #draw the axis
        cv2.line(frame, (cen_x, cen_y), (x2, y2), color, 2)
        cv2.putText(frame, label, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    #compute the centroid by averaging the end points of the axes
    centroid_x = int(centroid_x / axes_num)
    centroid_y = int(centroid_y / axes_num)

    #draw a circle at the centroid
    cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 255, 255), -1)

    #print(similarities)
    #show the frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
