"""This module containing a method to create segmentation masks from the 'annotation.json' (computed trough VIA).
It will create both 'man_seg*.tiff' and 'man_track*.tiff' for every original image from the '*.json' file
created trough VIA.

The method is inspired by https://github.com/maftouni/binary_mask_from_json/blob/main/binary_mask_from_json.py.
"""
from copy import deepcopy
import shutil
import json
import PIL.Image
import numpy as np
import os
import cv2
from typing import Dict, Union

from img_processing.imageUtils import visualize_image, visualize_mask

DEBUG_PATH = './tmp' # Folder for debug visualization
N_IMAGES = 2 # Number of images to print for the 


def debug_frames(file_names, d_images, d_drawed_images, d_masks, d_markers):
    
    # Clean and set up the debugging folder
    try:
        shutil.rmtree(DEBUG_PATH, ignore_errors=True)
    except Exception as e:
        print(f'Failed to delete directory: {e}')
    os.makedirs(DEBUG_PATH ,exist_ok=True) # Set up the debugging folder
    
    # Just visually debug the first 'N_IMAGES'
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

def create_masks_from_json(json_file: str, 
                           images_folder: str, 
                           seg_folder: str, 
                           pixels_limit: Union[int, float] = 7000, 
                           reducing_ratio: float = 0.2): 
    """
    
    
    
    Args:
        pixel_limit: needed for the adusting of the marker dimension (smaller marker in case of cells).
        reducing_ratio: Factor to reduce the initial marker dimension.
    """
    # 'pixel_limit' needed for the adusting of the marker dimension (smaller marker in case of cells)
    

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

    for name in all_file_names: # Loop over the original filenames - order of the '.json' keys mantained.

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
                shapes_x, shapes_y, id_obj = [], [], [] # In order to take into account EVs that compare/disappear it is added the 'id_obj' attribute.
                for segm_obj in data[name]['regions']: # Loop over the 'dict' (one for each segmented object)
                    id_obj.append(segm_obj['region_attributes']['name'])
                    shapes_x.append(segm_obj['shape_attributes']['all_points_x'])
                    shapes_y.append(segm_obj['shape_attributes']['all_points_y'])

            except :
                print(f"Possible different type of the dict keys ('*.json' file saved in the wrong format)")
                continue # TODO: Raise exception

            # Check to have all the segmented object with the 'name' filled
            if 'not_defined' in id_obj:
                print(f"Bug on id list: {id_obj}")
                raise ValueError(f"The list contains unexpected values: check that all the segmented object have a name assigned!")

            if len(set(id_obj)) < len(id_obj):
                print(f"Bug on id list: {id_obj}")
                raise ValueError(f"The list contains unexpected values: check that all the segmented object have a different 'name' assigned!")
            else:
                id_obj = [int(obj) for obj in id_obj]

            # Add the multiple stacked obj. on the same image
            multiple_ab = []
            for shape_x, shape_y in zip(shapes_x, shapes_y):
                multiple_ab.append(np.stack((shape_x, shape_y), axis=1)) # Annotated shape of one obj.

            # Kept for debug purpose.
            img2 = cv2.drawContours(img, multiple_ab, -1, (255,255,255), -1) # Multiple obj. on the same image
            d_drawed_images.append(img2)

            mask, track = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16), np.zeros((img.shape[0], img.shape[1]), dtype=np.uint16)
            
            print(f".. segmenting {len(multiple_ab)} elements ..")
            for idx, obj in enumerate(multiple_ab): # Add one image at the time to manage different color
                _ = cv2.drawContours(mask, [obj], -1, id_obj[idx], -1) # cv2.drawContours modify inplace AND return an image - Creation of the mask
                
                right_upper_limits = np.squeeze(obj.max(axis=0))
                left_lower_limits = np.squeeze(obj.min(axis=0))
                mean_point = np.mean([right_upper_limits, left_lower_limits] ,axis=0) # Get the central point of the shape (in my case the shape are all full rounds)

                # NOTE: My tracking marker is built as a square (4 corners) inside the shape.
                left_upper_corner = [int(np.mean([left_lower_limits[0], mean_point[0]])), int(np.mean([right_upper_limits[1], mean_point[1]]))]
                right_upper_corner = [int(np.mean([right_upper_limits[0], mean_point[0]])), int(np.mean([right_upper_limits[1], mean_point[1]]))]
                left_lower_corner = [int(np.mean([left_lower_limits[0], mean_point[0]])), int(np.mean([left_lower_limits[1], mean_point[1]]))]
                right_lower_corner = [int(np.mean([right_upper_limits[0], mean_point[0]])), int(np.mean([left_lower_limits[1], mean_point[1]]))]
                
                # Compute 'area' of the marker to see if it's a cell or a EVs.
                diff_x = right_upper_corner[0] - left_upper_corner[0]
                diff_y = right_upper_corner[1] - right_lower_corner[1]
                area = (diff_x) * (diff_y)
                print(f".. .. Element {id_obj[idx]} has area {area} .. ..")

                if area > pixels_limit: # Adjust the marker dimension in case the marker should be too big compared to the EVs

                    diff_x = int(diff_x * reducing_ratio) # 2/10 as fixed value for reducing the marker dim
                    diff_y = int(diff_y * reducing_ratio) # 2/10 as fixed value for reducing the marker dim
                    # Reducing on the x coord.
                    left_upper_corner[0] += diff_x
                    right_upper_corner[0] -= diff_x
                    left_lower_corner[0] += diff_x
                    right_lower_corner[0] -= diff_x

                    # Reducing on the y coord.
                    left_upper_corner[1] -= diff_y
                    right_upper_corner[1] -= diff_y
                    left_lower_corner[1] += diff_y
                    right_lower_corner[1] += diff_y
                
                square_marker = np.asarray([left_upper_corner, right_upper_corner, right_lower_corner, left_lower_corner]) # Order is important to construct the figure
                # NOTE: Id (color) propagated over time for the same obj -can't be assigned to different obj.
                _ = cv2.drawContours(track, [square_marker], -1, id_obj[idx], -1) # Creation of smaller objects corresponding to the current segmented cell - creation of the tracking mask.
                
            d_markers.append(track)
            d_masks.append(mask)
            
            mask_name = f'man_seg'+ image_name.split('.')[0].split('t')[-1] +'.tif' # Save the mask with the name of segmentation masks in the CTC datasets
            
            track_name = f'man_track'+ image_name.split('.')[0].split('t')[-1] +'.tif' # Save the tracking mask with the name of segmentation masks in the CTC datasets
            track_folder = seg_folder.split('SE')[0] + "TRA"
            
            if cv2.imwrite(os.path.join(seg_folder, mask_name), mask): # Save the segmentation mask
                print(f"Saved '{mask_name}'")
            else:
                print(f"Error when saving '{mask_name}' !")

            if cv2.imwrite(os.path.join(track_folder, track_name), track): # Save the segmentation tracking mask
                print(f"Saved '{track_name}'")
            else:
                print(f"Error when saving '{track_name}' !")

    # NOTE: Calling visual debug functions
    debug_frames(d_names, d_images, d_drawed_images, d_masks, d_markers)
    return None