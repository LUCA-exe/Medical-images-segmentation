"""main_img.py

This is the main executable file for running the processing of images functions.
"""

import os
from collections import defaultdict

class images_processor:

    def __init__(self, env, args):
        """Class to create a obj. that gather images signals from segmentation masks"""
        self.log = env['logger'] # Extract the needed evn. variables
        self.args = args # TODO: Select just the needed args
        self.images_folder = os.path.join(args.images_path, args.dataset) # Path in which the json will be saved
        #self.base_folder = 'images' # WARNING: Should be const

        self.log.info(f"'Image processor' object instantiated; working on '{self.images_folder}'")

    def set_dataset(self, new_images_path, new_dataset): # Change the target repository
        self.images_folder = os.path.join(new_images_path, new_dataset)

        return None

    def process_images(self):
        """ Load the images from a folder and compute the signals

        Args:
            split (str): What split to take in order to compute the images

        Returns:
            None 
    """
        data = {} # Dict to contain the different's data split

        folders = os.listdir(self.images_folder)
        self.log.info(f"Folder inside the dataset: {folders}")

        return None

    def __compute_signals(self, image_path, mask_path):
        """ Load the image and mask from the given path and compute the signals

        Args:
            split (str): What split to take in order to compute the images

        Returns:
            (dict): {'images_id':(str), 'signals_1': (float)}
    """

        images_path = os.path.join(self.images_folder, split, self.base_folder)
        images_files_path = os.listdir(images_path) # Gather the name of the images files
        
        self.log.info(f"Number of files readed in '{images_path}' is {len(images_files_path)}")
        self.log.debug(f"Images files read are : {images_files_path}")

        split_data = defaultdict(dict)

        for file_name in images_files_path:
            # gather all the signals for a specific image
            split_data[file_name] = __get_cr()
            
