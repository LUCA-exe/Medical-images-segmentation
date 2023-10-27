"""main_img.py

This is the main executable file for running the processing functions for the images/masks.
"""

import os
import numpy as np
import tifffile as tiff
import cv2
from collections import defaultdict
from img_processing.imageUtils import * # Remember to write the path the the 'importer' of this file is calling

class images_processor:

    def __init__(self, env, args, task='SEG'):
        """Class to create a obj. that gather images signals from segmentation masks"""

        self.log = env['logger'] # Extract the needed evn. variables
        self.args = args # TODO: Select just the needed args
        self.images_folder = os.path.join(args.images_path, args.dataset) # Path in which the json will be saved
        #self.base_folder = 'images' # WARNING: Should be const
        self.task = task # Final folder for the ground truth mask

        self.log.info(f"'Image processor' object instantiated: working on '{self.images_folder}'")
        
        os.makedirs('tmp', exist_ok=True) # Set up a './tmp' folder for debugging images
        self.debug_folder = 'tmp'
        
    def set_dataset(self, new_images_path, new_dataset): # Change the target repository
        self.images_folder = os.path.join(new_images_path, new_dataset)

        return None

    def collect_signals(self):
        """ Load the images from a folder and compute the signals

        Args:
            split (str): What split to take in order to compute the images

        Returns:
            None 
    """
        folders = os.listdir(self.images_folder)
        self.log.info(f"Folders inside the dataset: {folders}")

        masks_folders = [s for s in folders if s.endswith("GT")]
        self.log.debug(f"Folders with the masks: {masks_folders}")
        
        #masks_list = [] # List of lists containing masks path for each mask folder 
        
        for folder in masks_folders: # Main loop: for every folder gather the data

            current_images_path = os.path.join(self.images_folder, folder.split('_')[0]) # Fetch the original images from this folder
            current_path = os.path.join(self.images_folder, folder, self.task) # Compose the masks folder
            files_name = [s for s in os.listdir(current_path) if s.startswith("man")] # Get the file names of the masks
            #masks_list.append(files_name)
            files_name.sort()
            self.log.debug(f"Currently working in {current_path}: files are {files_name}")
            create_signals_file(self.log, current_path) # Prepare the '*.json' signals file to store the data

            stats = {} # Dict contatining 'id' and {'signal' : value}
            for file_name in files_name:
                current_mask_path = os.path.join(current_path, file_name)
                # For evey mask path, fetch the proper image path
                image_name = fetch_image_path(current_mask_path, current_images_path)
                current_image_path = os.path.join(current_images_path, image_name)

                # Ready to compute the signals for the coupled mask - image
                stats[image_name] = self.__compute_signals(current_image_path, current_mask_path) # Save the signals for every original image name

            # Finished to gather data from the current folder - update the existing '*.json'
            update_signals_file(self.log, current_path, stats)
            
        return None

    def __compute_signals(self, image_path, mask_path):
        """ Load the image and mask from the given paths and compute the signals

        Args:
            image_path (str): Path to load the '*.tif' image
            mask_path (str): Path to load the '*.tif' mask

        Returns:
            (dict): {'signals_1': (float), 'signals_2': (float)}
        """

        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)
        signals_dict = {}
        # TODO: Check if the mask has to be modified already here
        #print(f"Mask type: {type(mask)}")
        #print(f"Mask shape: {mask.shape}")
        #print(f"Few values of the mask {mask[:, :20]}")
        boolean_mask = mask != 0 # Create a boolean matrix

        obj_pixels = len(mask[boolean_mask]) # Pixels belonging to segmented objects considered
        tot_pixels = mask.shape[0] * mask.shape[1]
        print(f"Signal to noise ratio: {obj_pixels/tot_pixels * 100} %")
        #exit(1)
        #mask = np.ma.masked_array(mask, mask==0) # TODO: Check if the mask has to be modified already here
        
        
        # WORK IN PROGRESS: Init. segmenting the original to gather the signals
        #img2 = cv2.drawContours(img, multiple_ab, -1, (255,255,255), -1)

        # Visualization debug
        mask_name = os.path.basename(mask_path).split('.')[0]
        visualize_mask(mask, os.path.join(self.debug_folder, mask_name))
        #visualize_image(mask, os.path.join(self.debug_folder, mask_name))
        signals_dict['stn'] = obj_pixels/tot_pixels
        
        return signals_dict
            
