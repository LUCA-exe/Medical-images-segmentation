"""main_img.py

This is the main executable file for running the processing of images functions.
"""

from matplotlib import pyplot as plt
import numpy as np

# TODO: Save the images in tmp folder to visualize them

def visualize_raw_mask(mask):
    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    mask = np.ma.masked_array(mask, mask==0)
    axs[0].imshow(np.squeeze(mask)) # Raw mask
    axs[0].axis('off')
    axs[1].imshow(np.squeeze(mask), cmap='gray') # The segmentation map as a grey scale image
    axs[1].axis('off')
    fig.tight_layout()

def visualize_raw_res(image, mask):
    fig, axs = plt.subplots(1, 3, figsize=(12, 12))
    axs[0].imshow(np.squeeze(image)) # The visualization 'cmap' can be changed depending on the type of visualization for the 'raw' pixel value
    axs[0].axis('off')

    mask = np.ma.masked_array(mask, mask==0) # Use the built-in numpy class !
    axs[1].imshow(np.squeeze(mask)) # Raw mask
    axs[1].axis('off')

    axs[2].imshow(np.squeeze(mask), cmap='gray') # Gray scale
    axs[2].axis('off')

    fig.tight_layout()


# Util functions

def __load_image__(image_folder): # Util function of this class. Check if this pattern make sense
    """ Load the single channel or multiple channels of the required image.
        It return a dict with id and all the different images for each 'version' (only nuclei, only boundaries etc ..)

    Args:
        image_folder (str): Path to the different version of the image

    Returns:
        dict: {'id': image_folder, 'dapi': Image object, 'fitc': image object .. }
"""