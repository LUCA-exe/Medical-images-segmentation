"""imageUtils.py

This file contains utils function for the other scripts in the folder.
- Visualization for debug purpose
- Creation/updating of files containing images characteristics
- Debugging functions
"""

from matplotlib import pyplot as plt
import numpy as np
import os
from collections import defaultdict
import json
import tifffile as tiff


# Functions to visualize the images during processing
def visualize_mask(mask, file_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    
    mask = np.ma.masked_array(mask, mask==0)
    axs[0].imshow(np.squeeze(mask)) # Raw mask
    axs[0].axis('off')

    axs[1].imshow(np.squeeze(mask), cmap='gray') # The segmentation map as a grey scale image
    axs[1].axis('off')

    fig.tight_layout()
    plt.savefig(file_path)
    plt.close()

def visualize_image(image, file_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    
    axs[0].imshow(np.squeeze(image)) # Raw image
    axs[0].axis('off')

    axs[1].imshow(np.squeeze(image), cmap='gray') # Gray scale image
    axs[1].axis('off')

    fig.tight_layout()
    plt.savefig(file_path)
    plt.close()

def visualize_raw_res(image, mask, file_path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 12))
    axs[0].imshow(np.squeeze(image)) # The visualization 'cmap' can be changed depending on the type of visualization for the 'raw' pixel value
    axs[0].axis('off')

    mask = np.ma.masked_array(mask, mask==0) # Use the built-in numpy class !
    axs[1].imshow(np.squeeze(mask)) # Raw mask
    axs[1].axis('off')

    axs[2].imshow(np.squeeze(mask), cmap='gray') # Gray scale
    axs[2].axis('off')

    fig.tight_layout()
    plt.savefig(file_path)
    plt.close()


# Util function - called even from the 'utils.py' in the root folder
def log_image_characteristics(log, image_obj, name_obj):
    """ Log shape and max/min values of the images when downloading a new dataset (first check at the data)

    Args:
        image_obj (np.ndarray): Tensor of the image
        name_obj (str): Name to print on the debug string

    Returns:
        None
    """
    if name_obj == 'mask':
        log.debug(f"First {name_obj} analyzed > Shape: {image_obj.shape} Max/Min pixel values: {np.max(image_obj)}/{np.min(image_obj)} Different pixel values: {len(np.unique(image_obj))}")
    else:
        log.debug(f"First {name_obj} analyzed > Shape: {image_obj.shape} Max/Min pixel values: {np.max(image_obj)}/{np.min(image_obj)}")
    return None


def fetch_image_path(mask_path, images_folder):
    """ Given a path of a mask, return the corresponding image path

    Args:
        image_folder (str): Path to the different version of the image

    Returns:
        dict: {'id': image_folder, 'dapi': Image object, 'fitc': image object .. }
    """
    images_name = os.listdir(images_folder)
    for image_name in images_name: # Search for the correct image name

        p_name = image_name[1:]
        if p_name == os.path.basename(mask_path).split('seg')[-1]:
            # Proper image path found!
            return image_name
    
    return None # ERROR: Name not found!


def to_single_channel(image):
    """ Take an image and fuse the channels toghether

    Args:
        image (array): Numpu array obj. (expected of 3 dimensions)

    Returns:
        (array): Converted image
    """

    if (len(image.shape)) > 3:
        # TODO: Raise exeption!
        print(f"image has more than 3 dims!")
        return None

    else: # Consider all object in the image.
        return np.sum(image, axis=2) # Sum the pixels value along the last dim.
    
def create_signals_file(log, file_path, name='dataset_signals', ext='.json'): # Fow now don't pass files name as list, This function will retrieve them.
    """ Create a '*.json' file that contains signals gathered for every image.

    Args:
        log (obj): Object that handles the log messages
        file_path (str): Where to save the file
        name (str): Name of the file. The format will be '.json'
        ext (str)

    Returns:
        None
    """
    if os.path.exists(os.path.join(file_path, name + ext)): # Check if the file already exist, if already existing it contains already all the 'key'(images name).
        log.info(f"Signals file already existing in {file_path}, It will be not subscribed!")
        return None
    
    empty_dict = defaultdict(dict) # Just record all the images name to prepare to be filled later
    images_name = ['t' + s.split('seg')[-1] for s in os.listdir(file_path) if s.startswith("man")] # Load already the name of the images given the semented mask
    log.info(f"The signals file {name + ext} will be created in '{file_path}' for the images: {images_name}")
    
    [empty_dict[name] for name in images_name] # Fill the inital dict without signals

    with open(os.path.join(file_path, name + ext), "w") as outfile:
        json.dump(empty_dict, fp=outfile, indent = 4, sort_keys=True)
        log.info(f"File '{name + ext}' in '{file_path}' created correctly!")

    return None

# NOTE: It requires time complexity but It is modular: the information data can be filled in different times.
def update_signals_file(log, file_path, data, name='dataset_signals', ext='.json'):
    """ Read the '*.json' file and complement/insert the additional infomation for each images

    Args:
        log (obj): Object that handles the log messages
        file_path (str): Where to save the file
        data (dict): dict with singnals data that has to be updated/inserted for every image
        name (str): Name of the file. The format will be '.json'
        ext (str)

    Returns:
        None
    """
    f = open(os.path.join(file_path, name + ext))
    old_data = json.load(f) # Read the existing '*.json' to add new informations
    
    log.debug(f"File {name + ext} loaded from '{file_path}' loaded correctly!")

    for key, item in data.items(): # If there is a file, all the names are alredy inside it
        
        # 'item' contain the signals the have to be inserted
        for signals, value in item.items():
            old_data[key][signals] = value

    with open(os.path.join(file_path, name + ext), "w") as outfile:
        json.dump(old_data, fp=outfile, indent = 4, sort_keys=True)
        log.info(f"File updated correctly!")

    return None

# WARNING: The data will be saved in a file, if the '*.json' already exist It will be overwrited.
def save_aggregated_signals(log, file_path, data, name='aggreagated_signals', ext='.json'):
    """ Save the aggregated signals dict of a dataset in a '*.json'

    Args:
        log (obj): Object that handles the log messages
        file_path (str): Where to save the file
        data (dict): dict with signals data that has to be inserted
        name (str): Name of the file. The format will be '.json'
        ext (str)

    Returns:
        None
    """
    
    files = os.listdir(file_path)
    file_name = name + ext
    if file_name in files: # Check if the file already exist (just to notify)
        log.info(f"Aggregated file for the dataset {os.path.basename(file_path)} will be overwrite")

    with open(os.path.join(file_path, file_name), "w") as outfile:
        json.dump(data, fp=outfile, indent = 4, sort_keys=True)
        
        log.info(f"File {file_name} saved correctly in '{file_path}'!")

    return None


def aggregate_signals(log, signals_list, method='mean'):
    """ Create a dict that contains aggregated signals gathered for every image in the folders of a dataset.

    Args:
        signals_list (obj): List of dict; every dict contains the signals of a single image

    Returns:
        (dict): Return a unique dict of aggregated signals
    """

    # TODO: Upgrade and add functionalities: for now just compute the mean for every aggregated signals 
    total_dict = defaultdict(list)
    
    for single_dict in signals_list: # For every image metrics dict
        for key, value in single_dict.items(): # Append the value to the 'total_dict'
            total_dict[key].append(value)

    if method == 'mean': # If not applied any aggregation method the dict will contains a list of value for every metrics
        for k in total_dict.keys(): # Just aggregate with the chosen method
            total_dict[k] = np.mean(total_dict[k]) # Even if the defaultdict is set to list It works just with void key.

    log.debug(f"Signals of the current dataset aggregated correctly! (aggregation used: {method})")
    
    return total_dict # The obj. may differ in structure given the aggregation method


def debug_segmentation_masks(seg_masks_path):
    # Function to print seg_masks characteristics on console (just helps for creating the 'man_track.txt' in the 'TRA' folder)

    print(f"*** Printing masks properties on '{seg_masks_path}' ***")
    masks_ids = [mask for mask in os.listdir(seg_masks_path) if mask.endswith('.tif')]
    print(f"Mask found: {masks_ids}")

    labels_dict = defaultdict(list) # Dict containing {'element': [first_frame, last_frame]}, helps the manual creation of 'man_track.txt'.
    for mask_id in masks_ids:
        mask = tiff.imread(os.path.join(seg_masks_path, mask_id))

        print(f"Mask {mask_id}: shape {mask.shape}")
        unique_values = np.unique(mask)
        
        for value in unique_values: # Remember, the 'value' representing the object segmented is the pixel value assigned to the object along the time lapse
            labels_dict[value].append(int(mask_id.split('.')[0].split('g')[-1])) # Append just the frame int value if it is present the 'label' key object inside the current mask

    print(f"The columns are: Label - first frame - last frame (the print is below to facilitate the '*.txt' creation)")
    for key, frame_list in labels_dict.items():

        if key == 0: # Avoid the background label
            continue

        print(f"{key} {min(frame_list)} {max(frame_list)}")

    print("*** Debug maks properties complete! ***")

    return None