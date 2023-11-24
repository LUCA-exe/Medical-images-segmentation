"""create_mask.py

This is a 'util' file to create segmentation masks from the 'annotation.json' (computed trough VIA) and the following 'man_trackT.tif'.
(It is separate from the general repository, just a script that you can call from a notebook cell).
"""

from copy import deepcopy
import shutil
import json
import PIL.Image
import numpy as np
import os
import cv2

from img_processing.imageUtils import visualize_image, visualize_mask

DEBUG_PATH = './tmp' # Folder for debug visualization
N_IMAGES = 2 # Number of images to debug visually


# TODO: Add debug visualization of the first and last frames (import functions from imageUtils)
def debug_frames(file_names, d_images, d_drawed_images, d_masks, d_markers):
    
    # Clean and set up the debugging folder
    try:
        shutil.rmtree(DEBUG_PATH)
    except Exception as e:
        print(f'Failed to delete directory: {e}')
    os.makedirs(DEBUG_PATH ,exist_ok=True) # Set up the debugging folder
    
    for i in range(N_IMAGES):

        name = file_names[i]
        img = d_images[i]
        drawed_images = d_drawed_images[i]
        mask = d_masks[i]
        marker = d_markers[i]

        visualize_image(img, os.path.join(DEBUG_PATH, name))
        visualize_image(drawed_images, os.path.join(DEBUG_PATH, f"drawed_{name}"))
        visualize_mask(mask, os.path.join(DEBUG_PATH, f"man_seg{name.split('t')[-1]}"))
        visualize_mask(marker, os.path.join(DEBUG_PATH, f"man_track{name.split('t')[-1]}"))
    
    print(f"The {N_IMAGES} images annotation process is shown on the {DEBUG_PATH}!")

    return None


# Reference to the original repository (https://github.com/maftouni/binary_mask_from_json/blob/main/binary_mask_from_json.py)
def create_masks_from_json(json_file, images_folder, target_folder):

    print("*** Creation of the segmentation masks ***")
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

    d_names, d_images, d_drawed_images, d_masks, d_markers = [], [], [], [], []  

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

            # Kept for debug purpose.
            img2 = cv2.drawContours(img, multiple_ab, -1, (255,255,255), -1) # Multiple obj. on the same image
            d_drawed_images.append(img2)

            mask, track = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16), np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
            
            for idx, obj in enumerate(multiple_ab): # Add one image at the time to manage different color
                _ = cv2.drawContours(mask, [obj], -1, idx+1, -1) # cv2.drawContours modify inplace AND return an image - Creation of the mask
                
                right_upper_limits = np.squeeze(obj.max(axis=0))
                left_lower_limits = np.squeeze(obj.min(axis=0))
                mean_point = np.mean([right_upper_limits, left_lower_limits] ,axis=0) # Get the central point of the shape (in my case the shape are all full rounds)
                
                #print(right_upper_limits)
                #print(left_lower_limits)
                #print(mean_point)

                

                # NOTE: My tracking marker is built as a square (4 corners) inside the shape.
                left_upper_corner = [int(np.mean([left_lower_limits[0], mean_point[0]])), int(np.mean([right_upper_limits[1], mean_point[1]]))]
                right_upper_corner = [int(np.mean([right_upper_limits[0], mean_point[0]])), int(np.mean([right_upper_limits[1], mean_point[1]]))]
                left_lower_corner = [int(np.mean([left_lower_limits[0], mean_point[0]])), int(np.mean([left_lower_limits[1], mean_point[1]]))]
                right_lower_corner = [int(np.mean([right_upper_limits[0], mean_point[0]])), int(np.mean([left_lower_limits[1], mean_point[1]]))]
                square_marker = np.asarray([left_upper_corner, right_upper_corner, right_lower_corner, left_lower_corner]) # Order is important to construct the figure
                
                #print(f"Round shape: {obj}")
                #print(f"Square marker: {square_marker}")
                #exit(1)
                _ = cv2.drawContours(track, [square_marker], -1, idx+1, -1) # Creation of smaller objects corresponding to the current segmented cell - creation of the tracking mask

                
            d_markers.append(track)
            d_masks.append(mask)
            
            mask_name = f'man_seg'+ image_name.split('.')[0].split('t')[-1] +'.tif' # Save the mask with the name of segmentation masks in the CTC datasets
            
            if cv2.imwrite(os.path.join(target_folder, mask_name), mask): # Save the segmentation mask
                print(f"Saved '{mask_name}'")
            else:
                print(f"Error when saving '{mask_name}' !")

    # Call visual debug functions
    debug_frames(d_names, d_images, d_drawed_images, d_masks, d_markers)

    return None