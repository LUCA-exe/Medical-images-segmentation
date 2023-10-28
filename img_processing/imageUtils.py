"""main_img.py

This is the main executable file for running the processing of images functions.
"""

from matplotlib import pyplot as plt
import numpy as np
import os
from collections import defaultdict
import json


# Save the images in tmp folder to visualize them
def visualize_mask(mask, file_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    
    mask = np.ma.masked_array(mask, mask==0)
    axs[0].imshow(np.squeeze(mask)) # Raw mask
    axs[0].axis('off')

    axs[1].imshow(np.squeeze(mask), cmap='gray') # The segmentation map as a grey scale image
    axs[1].axis('off')

    fig.tight_layout()
    plt.savefig(file_path)

def visualize_image(image, file_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    
    axs[0].imshow(np.squeeze(image)) # Raw image
    axs[0].axis('off')

    axs[1].imshow(np.squeeze(image), cmap='gray') # Gray scale image
    axs[1].axis('off')

    fig.tight_layout()
    plt.savefig(file_path)

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


# Util functions

def __load_image__(image_folder): # Util function of this class. Check if this pattern make sense
    """ Load the single channel or multiple channels of the required image.
        It return a dict with id and all the different images for each 'version' (only nuclei, only boundaries etc ..)

    Args:
        image_folder (str): Path to the different version of the image

    Returns:
        dict: {'id': image_folder, 'dapi': Image object, 'fitc': image object .. }
    """


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
    else:
        return np.sum(image, axis=2) # Sum the pixels value along the last dim
    



def create_signals_file(log, file_path, name='dataset_signals', ext='.json'): # Fow now don't pass files name as list, This function will retrieve them
    """ Create a '*.json' file that contains signals gathered for every image.

    Args:
        log (obj): Object that handles the log messages
        file_path (str): Where to save the file
        name (str): Name of the file. The format will be '.json'
        ext (str)

    Returns:
        None
    """
    if os.path.exists(os.path.join(file_path, name + ext)):
        log.info(f"Signals file already existing in {file_path}, It will be not subscribed!")
        return None
    
    empty_dict = defaultdict(dict) # Just record all the images name to prepare to be filled later
    images_name = ['t' + s.split('seg')[-1] for s in os.listdir(file_path) if s.startswith("man")] # Load already the name of the images given the semented mask
    log.info(f"The signals file {name + ext} will be created in {file_path} for the images: {images_name}")
    
    [empty_dict[name] for name in images_name] # Fill the inital dict without signals

    with open(os.path.join(file_path, name + ext), "w") as outfile:
        json.dump(empty_dict, fp=outfile, indent = 4, sort_keys=True)
        log.info(f"File '{name + ext}' in '{file_path}' created correctly!")

    return None

# NOTE: It requires time complexity but It is modular: the information data can be filled in different times
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

    
     


