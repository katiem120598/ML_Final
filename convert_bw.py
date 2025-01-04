#convert all of the images in all of the subfolders in the "bw_training" folder to black and white
#____________________________________________________________________________________
#____________________________________________________________________________________
import os
from PIL import Image
import numpy
import cv2

import os
from PIL import Image

def convert_to_bw(input_folder):
    # Walk through all files and directories within the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Check if the file is an image (add or remove extensions as needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                # Construct full file path
                full_file_path = os.path.join(root, file)
                # Open the image file
                with Image.open(full_file_path) as img:
                    # Convert the image to grayscale
                    bw = img.convert('L')
                    # Save the grayscale image back to disk
                    bw.save(full_file_path)
                print(f'Converted {full_file_path} to black and white.')

if __name__ == "__main__":
    # Path to the directory containing the "bw_training" folder
    input_folder = 'bw_training'
    convert_to_bw(input_folder)
    print("All images have been converted to black and white.")
