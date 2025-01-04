from fragment_analysis_revised import fragment_analyze
import os

#find the number of folders in the training_data folder
#model_list = ['classifier_model']
model_list = ['classifier_model','classifier_model_bw','classifier_model_small']
#create a list of the names of the files in the detail_images folder
detail_images_folder = os.path.join(os.getcwd(), 'detail_images')

# Create a list of the names of the files in the detail_images folder
file_names = [f for f in os.listdir(detail_images_folder) if os.path.isfile(os.path.join(detail_images_folder, f))]

CURRENT_PATH = os.getcwd()
IMAGES_FOLDER = os.path.join(CURRENT_PATH, r'detail_images')
#MODEL_FOLDER = os.path.join(CURRENT_PATH, 'classifier_model')
#TEST_FOLDER = os.path.join(CURRENT_PATH, r'testing_data')

for model in model_list:
    for i in range (0,5):
        image = file_names[i]
        fragment_analyze(image,model,i+1,False)
        fragment_analyze(image,model,i+1,True)