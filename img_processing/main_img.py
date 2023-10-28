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
        self.thr = args.cell_dim

        os.makedirs('tmp', exist_ok=True) # Set up a './tmp' folder for debugging images
        self.debug_folder = 'tmp'

        self.log.info(f"'Image processor' object instantiated: working on '{self.images_folder}'")
        
        
        
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
        # This function will process images; cleaning the './tmp' folder
        if os.listdir(self.debug_folder): # Debug dir has some files
            print("folder not empty, removing files")
            for file_name in os.listdir(self.debug_folder):
                os.remove(file_name)
             
        # Start collecting the folders path for the current dataset
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
            self.log.debug(f"Currently working in '{current_path}': files are {files_name}")
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

    # TODO: Working for gather signal both from my dataset and others (e.g. CTC/DSB 2018 datasets)
    def __compute_signals(self, image_path, mask_path):
        """ Load the image and mask from the given paths and compute the signals

        Args:
            image_path (str): Path to load the '*.tif' image
            mask_path (str): Path to load the '*.tif' mask

        Returns:
            (dict): {'signals_1': (float), 'signals_2': (float)}
        """

        image = tiff.imread(image_path)
        if (len(image.shape) > 2):
            image = to_single_channel(image) # Convert the image if needed
            self.log.debug(f".. image '{os.path.basename(image_path)}' converted to one channel ..")

        mask = tiff.imread(mask_path)
        signals_dict = {} # Init the dict for the signals

        
        boolean_mask = mask != 0 # Create a boolean matrix

        obj_pixels = len(mask[boolean_mask]) # Pixels belonging to segmented objects considered
        tot_pixels = mask.shape[0] * mask.shape[1] # Total pixels of the image

        

        background_pixels = image[~boolean_mask] # Take the pixel not belonging to the segmented objects
        print(f"Rumore nel background: {np.std(background_pixels)}")


        # Debug purpose
        print(f"Numero di oggetti (incluso il background): {len(np.unique(mask))}")
        print(f"Valori degli oggetti (incluso background): {np.unique(mask)}")

        # TODO: Make this adjusted to the current dataset.. my images have different number of pixels
        #thr = 7000 # For now less than 7000 pixels is an EVs for sure (keeping into account that it depends on the image quality)
        mean_cells, std_cells = [], [] # Mean and std inside the segmented cells
        obj_values = np.unique(mask) # Store the pixel value for each object
        obj_dims = [np.count_nonzero(mask==value) for value in obj_values] # Numbers of pixels occupied for every objects (in order of total pixels)
        
        for value, count in zip(obj_values, obj_dims): # For now less than 7000 pixels is an EVs for sure (keeping into account that it depends on the image quality)
            # DEBUG - tried this on other datasets
            print(f"   {value} - {count}")

            if count > self.thr: # The current obj. is a cell or background (background will occupy the FIRST position in the lists)
                current_mask = mask == value
                current_pixels = image[current_mask] # Take from the original image the pixels value corresponding to the current object mask
                mean_cells.append(np.mean(current_pixels)) 
                std_cells.append(np.std(current_pixels))

        print("-------")

        #mask = np.ma.masked_array(mask, mask==0) # TODO: Check if the mask has to be modified already here

        # Visualization debug - both image/mask
        mask_name = os.path.basename(mask_path).split('.')[0]
        visualize_mask(mask, os.path.join(self.debug_folder, mask_name))
        image_name = os.path.basename(image_path).split('.')[0]
        visualize_image(image, os.path.join(self.debug_folder, image_name))
        
        signals_dict['cc'] = np.mean(mean_cells[1:]) # Cell color: mean of the cells pixels
        signals_dict['cv'] = np.mean(std_cells[1:]) # Cell variations: mean of the cells pixels std
        signals_dict['stn'] = obj_pixels/tot_pixels # Signal to noise ratio
        signals_dict['bn'] = std_cells[0] # Background noise - computed during the stats over the segmented obj.

        return signals_dict
            
