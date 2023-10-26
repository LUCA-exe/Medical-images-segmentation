"""main_img.py

This is the main executable file for running the processing of images functions.
"""

import os
from collections import defaultdict
from imageUtils.py import *

class images_processor:

    def __init__(self, env, args, task='SEG'):
        """Class to create a obj. that gather images signals from segmentation masks"""

        self.log = env['logger'] # Extract the needed evn. variables
        self.args = args # TODO: Select just the needed args
        self.images_folder = os.path.join(args.images_path, args.dataset) # Path in which the json will be saved
        #self.base_folder = 'images' # WARNING: Should be const
        self.task = task # Final folder for the ground truth mask

        self.log.info(f"'Image processor' object instantiated; working on '{self.images_folder}'")
        
        
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
        
        for folder in masks_folders: # Main loop; for every folder gather the data

            current_images_path = os.path.join(self.images_folder, folder.split('_')[0]) # Fetch the original images from this folder
            current_path = os.path.join(self.images_folder, folder, self.task) # Compose the masks folder
            files_name = [s for s in os.listdir(current_path) if s.startswith("man")] # Get the file names of the masks
            #masks_list.append(files_name)
            files_name.sort()
            self.log.debug(f"{files_name}")

            stats = {} # Dict contatining 'id' and {'signal' : value}
            for file_name in files_name:
                current_mask_path = os.path.join(current_path, file_name)
                # For evey mask path, fetch the proper image path
                image_name = fetch_image_path(current_mask_path, current_images_path)
                current_image_path = os.path.join(current_images_path, image_name)

                # Ready to compute the signals for the coupled mask - image
                self.__compute_signals(current_image_path, current_mask_path)

        return None

    def __compute_signals(self, image_path, mask_path):
        """ Load the image and mask from the given paths and compute the signals

        Args:
            split (str): What split to take in order to compute the images

        Returns:
            (dict): {'images_id':(str), 'signals_1': (float)}
    """

        '''images_path = os.path.join(self.images_folder, split, self.base_folder)
        images_files_path = os.listdir(images_path) # Gather the name of the images files
        
        self.log.info(f"Number of files readed in '{images_path}' is {len(images_files_path)}")
        self.log.debug(f"Images files read are : {images_files_path}")

        split_data = defaultdict(dict)

        for file_name in images_files_path:
            # gather all the signals for a specific image
            split_data[file_name] = __get_cr()'''
        
        return None
            
