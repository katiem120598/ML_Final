from classifier_helper import *
import os
import pandas as pd
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#_____________________________________________________________
#_____________________________________________________________
#visualize the images using a 2D projection with the XY coordinates
#corresponding to the activations along two classes
#_____________________________________________________________
#_____________________________________________________________
def classifier_visualizer(class1_idx,class2_idx,folder_path,type):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    CURRENT_PATH = os.getcwd()
    MODEL_FOLDER = os.path.join(CURRENT_PATH, 'classifier_model')
    IMAGES_FOLDER = os.path.join(CURRENT_PATH,folder_path)
    OUTPUT_FOLDER = os.path.join(CURRENT_PATH, 'timeline_images')

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = Classifier.createWithImgNetModel(MODEL_FOLDER, device=DEVICE)
    classifier.load()

    def is_image_file(file:str) -> bool:
        return file.lower().endswith(".jpg") or file.lower().endswith(".png") or file.lower().endswith(".jpeg")

    image_files = [os.path.join(IMAGES_FOLDER, f) for f in os.listdir(IMAGES_FOLDER) if is_image_file(f)]

    class_1 = class1_idx
    class_2 = class2_idx

    axis1 = classifier.class_names[class_1]
    axis2 = classifier.class_names[class_2]

    print(f"Axis 1: {axis1}")
    print(f"Axis 2: {axis2}")


    x_data = []
    y_data = []

    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')

        activation, probability, maximum_index = classifier.classify(image)

        x_data.append(activation[class_1])
        y_data.append(activation[class_2])

    x_min = min(x_data)
    x_max = max(x_data)
    y_min = min(y_data)
    y_max = max(y_data)

    #calculate an estimate size of the image thumbnails based on the range of the data
    img_sz = (x_max - x_min) * 0.04

    import matplotlib.pyplot as plt

    bg_color = (0.35, 0.45, 0.6)
    axes_color = (0.5, 1.0, 1.0)
    grid_color = (0.2, 0.6, 0.7)
    text_color = (0.8, 1.0, 1.0)
    text_bg_color = (0.1, 0.55, 0.6, 0.4)
    dot_color = (1.0, 0.0, 1.0)

    """
    plt.figure(figsize=(6,6), facecolor=bg_color)
    plt.title(f"Planar projection along {axis1} and {axis2}")
    ax = plt.gca()


    ax.set_facecolor(bg_color)
    ax.axhline(0, color=axes_color)
    ax.axvline(0, color=axes_color)
    plt.xlabel(axis1)
    plt.ylabel(axis2)

    plt.xlim(x_min-img_sz, x_max+img_sz)
    plt.ylim(y_min-img_sz, y_max+img_sz)
    """
    df = pd.DataFrame(columns=['year', 'difference','solo'])
    if not os.path.exists('activation_results'):
        os.makedirs('activation_results')


    plt.figure(figsize = (20,10))
    plt.scatter(df['year'], df['difference'], c = 'blue', s = 10)
    #loop through the image fiels to create and display a thumbnail of each image
    for i, img_file in enumerate(image_files):
        #load the image
        image = Image.open(img_file)

        #resize to a smaller size for better performance
        image = image.resize((128, 128), resample=Image.LANCZOS)
        
        #display
        x = x_data[i]
        y = y_data[i]
        print(os.path.basename(img_file).split('_')[1]+": "+f"{y_data[i]-x_data[i]} , {y_data[i]}")
        year = int(os.path.basename(img_file).split('_')[1])
        diff = y_data[i]-x_data[i]
        solo = y_data[i]
        #output a df and save to csv with year in one column and difference in the other
        new_row = {'year': year, 'difference': diff, 'solo': y_data[i]}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        image_thumbnail = image.copy()
        image_thumbnail.thumbnail((60, 60), Image.LANCZOS)
        ax = plt.gca()
        im = OffsetImage(image_thumbnail, zoom=1)
        if type == 'difference':
            ab = AnnotationBbox(im, (year, diff), frameon=False, box_alignment=(0.5, 0.5))
        else:
            ab = AnnotationBbox(im, (year, solo), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
        #make 'activation_results' folder if it doesn't exist
        


        #plot the image centered at the x, y coordinates
        #ax.imshow(image, extent=(x - img_sz, x + img_sz, y - img_sz, y + img_sz), aspect='auto', alpha = 0.9, cmap='gray')
        #label the image with the file name up to the second underscore
        #plt.text(x, y, os.path.basename(img_file).split('_')[1], color=text_color, backgroundcolor=text_bg_color, ha='center', va='center')
    style_name = os.path.basename(IMAGES_FOLDER).split('_')[1]
    if type == 'difference':
        plt.title(f'{style_name} vs. International Style Timeline')
    else:
        plt.title(f'{style_name} Timeline')
    if type == 'difference':
        plt.ylabel(f'<-- International Style                                                        {style_name} -->')
    else:
        plt.ylabel(f'<-- Less {style_name}                                                        More {style_name} -->')
        
    plt.grid(False)
    plt.xlim(1800,2020)
    plt.ylim(-20,20)
    #make x=0 a bold black line
    plt.axhline(0, color='black', linewidth=.5)
    #plot a dotted line from the center of each image to the x=0 line
    if type == 'difference':
        plt.vlines(df['year'], df['difference'],linestyles = "dashed",color = "k", ymax = 0, linewidth=0.5)
    else:
        plt.vlines(df['year'], df['solo'],linestyles = "dashed",color = "k", ymax = 0, linewidth=0.5)
    # Move the x-axis ticks to the center at y=0
    ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['top'].set_color('none')
    xticks = np.arange(1800, 2021, 20)
    ax.set_xticks(xticks[1:])
    # Remove the y-ticks
    ax.set_yticks([])

    # Remove the left and right spines (borders)
    ax.spines['right'].set_color('none')
    # Save plot to OUTPUT_FOLDER with increased resolution
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{style_name}_{type}.png'), dpi=300)
    #plt.show()
    #save plot to OUTPUT_FOLDER
    

    return

classifier_visualizer(0,1,'testing_data/B_European_testing','difference')
classifier_visualizer(0,1,'testing_data/B_European_testing','solo')