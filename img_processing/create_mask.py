"""create_mask.py

This is a 'util' file to create segmentation masks from the 'annotation.json' (computed trough VIA).
(It is separate from the general repository, just a script that you can call from a notebook cell).
"""

from copy import deepcopy
import json
import PIL.Image
import numpy as np
import os
import cv2

from img_processing.imageUtils import visualize_image, visualize_mask

DEBUG_PATH = './tmp' # Folder for debug visualization
N_IMAGES = 2 # Number of images to debug visually


# TODO: Add debug visualization of the first and last frames (import functions from imageUtils)
def debug_frames(file_names, d_images, d_drawed_images, d_masks):

    os.makedirs(DEBUG_PATH ,exist_ok=True) # Set up the debugging folder
    
    for i in range(N_IMAGES):

        name = file_names[i]
        img = d_images[i]
        drawed_images = d_drawed_images[i]
        mask = d_masks[i]

        visualize_image(img, os.path.join(DEBUG_PATH, name))
        visualize_image(drawed_images, os.path.join(DEBUG_PATH, f"drawed_{name}"))
        visualize_mask(mask, os.path.join(DEBUG_PATH, f"man_seg{name.split('t')[-1]}"))
    
    print(f"The {N_IMAGES} images annotation process is shown on the {DEBUG_PATH}!")


# Reference to the original repository (https://github.com/maftouni/binary_mask_from_json/blob/main/binary_mask_from_json.py)
def create_mask_from_json(json_file, images_folder, target_folder):

    print("*** Creation of the segmentation mask ***")
    print(f"Working on {json_file} ..")

    with open(json_file, 'r') as read_file:
        data = json.load(read_file)

    all_file_names=list(data.keys())
    print(f"Read {len(all_file_names)} names from the '*.json' file:  {all_file_names}")

    images_names = []
    # Gather the name of the images to segment - can use 'listdir'
    for root, dirs, files in os.walk(images_folder, topdown=True):
        print(f"Reading images in '{root}'")
        
        for filename in files:
            print(f".. Reading {filename} ..")
            images_names.append(filename)

    d_names, d_images, d_drawed_images, d_masks = [], [], [], []  

    for name in all_file_names: # Loop over the original filenames - order of the '.json' keys mantained

        image_name = data[name]['filename'] # Extract the current 'real' images filename 
        d_names.append(image_name.split('.')[0])

        print(f".. Working on image {image_name} ..")
        image_name = image_name.split('.')[0] + '.tif' # Original name.format of the current image

        if image_name in images_names: # Change the extension of the image (from '.jpg' to '.tif')
            img = np.asarray(PIL.Image.open(os.path.join(images_folder, image_name)))

            d_images.append(deepcopy(img))
        else:
            print(">> Exception! Pass to another the image ..")
            continue # Can't open the current image

        if data[name]['regions'] != {}: # If there are annotaions
            try:
                shapes_x, shapes_y = [], []
                for segm_obj in data[name]['regions']: # Loop over the 'dict' (one for each segmented object)
                    shapes_x.append(segm_obj['shape_attributes']['all_points_x'])
                    shapes_y.append(segm_obj['shape_attributes']['all_points_y'])

            except :
                print(f"Possible different type of the dict keys ('*.json' file saved in the wrong format)")
                continue # TODO: Raise exception

            # Add the multiple stacked obj. on the same image
            multiple_ab = []
            for shape_x, shape_y in zip(shapes_x, shapes_y):
                multiple_ab.append(np.stack((shape_x, shape_y), axis=1)) # Annotated shape of one obj.

            # NOTE: Every obj. has to have different colour, and the same obj. along the frames has to have the same colour
            #colour_span = floor(255/len(multiple_ab)) # 255 is the maximum value of the pixel
            #colours = [colour for colour in range(colour_span, 256, colour_span)] # list comprehension

            # Kept for debug purpose.
            img2 = cv2.drawContours(img, multiple_ab, -1, (255,255,255), -1) # Multiple obj. on the same image
            d_drawed_images.append(img2)

            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
            
            for idx, obj in enumerate(multiple_ab): # Add one image at the time to manage different color
                _ = cv2.drawContours(mask, [obj], -1, idx+1, -1) # cv2.drawContours modify inplace AND return an image
            d_masks.append(mask)
            
            mask_name = f'man_seg'+ image_name.split('.')[0].split('t')[-1] +'.tif' # Save the mask with the name of segmentation masks in the CTC datasets
            
            if cv2.imwrite(os.path.join(target_folder, mask_name), mask): # Save the segmentation mask
                print(f"Saved '{mask_name}'")
            else:
                print(f"Error when saving '{mask_name}' !")

    # Call visual debug functions
    debug_frames(d_names, d_images, d_drawed_images, d_masks)

    return None