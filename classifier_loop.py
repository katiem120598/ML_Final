from classifier_use_visualization_2axes import *
import os

#find the number of folders in the training_data folder
CURRENT_PATH = os.getcwd()
IMAGES_FOLDER = os.path.join(CURRENT_PATH, r'training_data')
MODEL_FOLDER = os.path.join(CURRENT_PATH, 'classifier_model')
TEST_FOLDER = os.path.join(CURRENT_PATH, r'testing_data')

#find the number of folders in the training_data folder
class_folders = [f for f in os.listdir(IMAGES_FOLDER) if os.path.isdir(os.path.join(IMAGES_FOLDER, f))]
test_folders = [f for f in os.listdir(TEST_FOLDER) if os.path.isdir(os.path.join(TEST_FOLDER, f))]
class_count = len(class_folders)

#cycle through all the folders in testing_data
for i in range(1,len(class_folders)):
    folder_path = os.path.join(TEST_FOLDER, test_folders[i-1])
    classifier_visualizer(0,i,folder_path,'difference')
    classifier_visualizer(0,i,folder_path,'solo')
