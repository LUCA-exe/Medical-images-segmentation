"""main_img.py

This is the main executable file for running the processing of images functions.
"""

import os
from collections import defaultdict

class images_processor:

    def __init__(self, env, args):

        self.log = env['logger'] # Extract the needed evn. variables
        self.args = args # TODO: Select just the needed args
        self.images_folder = os.path.join(args.images_path, args.images_mode) # Path in which the json will be saved
        self.base_folder = 'images' # WARNING: Should be const

        self.log.info(f"'Image processor' object instantiated; working on '{self.images_folder}'")


    def process_images(self, split='train'): # TODO: Split can be both parser/config file
        """ Load the images from a single split or multiple and compute for every one Its signals/properties

        Args:
            split (str): What split to take in order to compute the images

        Returns:
            None 
    """
        data = {} # Dict to contain the different's data split

        if split == 'all': # Compute all the signals for different splits
            self.__compute_signals('train')
            self.__compute_signals('val')
            self.__compute_signals('test')
        else:
            self.__compute_signals(split)

        return None

    def __compute_signals(self, split):
        """ Load the images from a single split and compute for every one Its signals/properties

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
            




    # Util functions

    def __load_image__(image_folder): # Util function of this class. Check if this pattern make sense
        """ Load the single channel or multiple channels of the required image.
            It return a dict with id and all the different images for each 'version' (only nuclei, only boundaries etc ..)

        Args:
            image_folder (str): Path to the different version of the image

        Returns:
            dict: {'id': image_folder, 'dapi': Image object, 'fitc': image object .. }
    """
