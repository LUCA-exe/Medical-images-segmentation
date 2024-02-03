"""main_img.py

This is the main executable file for running the analysis and plotting functions for the 
images and masks.
"""

from copy import deepcopy
import json
from math import floor
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff
import cv2
from collections import defaultdict
from img_processing.imageUtils import *

# TODO: Move to a .env file
TEMPORARY_FOLDER = "./tmp"
VISUALIZATION_FOLDER = "./results"

class images_processor:

    def __init__(self, env, args, dataset, thresholds, names=['All cells'], task='SEG'):
        """Class to create an obj. that gather images signals from segmentation masks"""

        self.log = env['logger'] # Extract the needed env. variables - log
        self.args = args
        self.images_folder = os.path.join(args.train_images_path, dataset) # Path in which the json will be saved
        self.task = task # Final folder for the ground truth mask.
        self.considered_images = args.max_images # Set a limit to the number of masks to use.

        self.names = names # In case of '--split_signals' True, the names will be a list of multiple values instead of one
        self.thresholds = thresholds # Thresholds to consider for gathering the cell statistics - list of list.

        os.makedirs(TEMPORARY_FOLDER, exist_ok=True) # Set up a './tmp' folder for debugging images processing
        self.debug_folder = TEMPORARY_FOLDER

        self.log.info(f"'Image processor' object instantiated: working on '{self.images_folder}'")
        
        
    def set_dataset(self, new_images_path, new_dataset): # Change the target repository
        self.images_folder = os.path.join(new_images_path, new_dataset)
        self.log.info(f"Dataset considered by the processor object changed to {new_dataset}!" )
        return None

    # TODO: Verifiy that 'perc_pixels' should be a parameters callable from the main and NOT a paramereter on the program
    def collect_signals(self, perc_pixels = 0.2):
        """ Load the images from a folder and compute the signals

        Args:
            perc_pixels (float): Percent value used for a metric computation in the images (in particular for '')

        Returns:
            None 
        """
        self.log.info(f"Collecting properties (max {self.considered_images} for every masks folders) of '{self.images_folder}' ..")

        # This function will process images: cleaning the './tmp' folder first.
        if os.listdir(self.debug_folder): # 'Debug dir' has some files: delete all
            self.log.debug(f"Folder '{self.debug_folder}' contains files, It will be cleaned ..")
            shutil.rmtree(self.debug_folder)
            os.makedirs(self.debug_folder, exist_ok=True)
             
        # Start collecting the folders path for the current dataset
        folders = os.listdir(self.images_folder)
        dataset_name = self.images_folder.split('/')[-1]
        self.log.info(f"Folders inside the dataset ({dataset_name}): {folders}")

        masks_folders = [s for s in folders if s.endswith("GT")]
        self.log.debug(f"Folders with the masks: {masks_folders}")
        
        total_stats = defaultdict(list) # List of dict values containing all the computed signals - It is  a dict with number keys equal to the 'self.names'.
        
        if len(self.names) > 1:
            self.log.debug(f"Cells thresholds considered for {self.names}:  {self.thresholds}")

        for folder in masks_folders: # Main loop: for every folder gather the data.

            for threshold, name in zip(self.thresholds, self.names):
                
                current_images_path = os.path.join(self.images_folder, folder.split('_')[0]) # Fetch the original images from this folder.
                current_path = os.path.join(self.images_folder, folder, self.task) # Compose the masks folder.

                files_name = [s for s in os.listdir(current_path) if s.startswith("man")] # Get the file names of the masks.
                files_name.sort()
                self.log.debug(f"Currently working in '{current_path}': files are {files_name}")

                # Componing the current file name
                signals_file_name = dataset_name + ' ' + name
                create_signals_file(self.log, current_path, name = signals_file_name) # Prepare the '*.json' signals file to store the data
                
                stats = {} # Dict contatining 'id' and {'signal' : value} of masks of a single folder
                if len(files_name) > self.considered_images: files_name = files_name[:self.considered_images] # Consider just a limited number of masks for current folders
                
                for file_name in files_name:
                    current_mask_path = os.path.join(current_path, file_name)
                    image_name = fetch_image_path(current_mask_path, current_images_path) # For evey mask path, fetch the proper image path (images masks less than total of the images)
                    current_image_path = os.path.join(current_images_path, image_name)

                    # Ready to compute the signals for the coupled mask - image
                    self.log.debug(f".. working on {image_name} - {file_name} ..")
                    stats[image_name] = self.__compute_signals(current_image_path, current_mask_path, perc_pixels, threshold) # Save the signals for every original image name

                    total_stats[name].append(deepcopy(stats[image_name])) # Store the image signals for the future aggregation of this dataset (copy, not the reference)

                # Finished to gather data from the current folder - update the existing '*.json' (on the current folder)
                update_signals_file(self.log, current_path, stats, name = signals_file_name)

        for name in self.names:
            self.log.info(f".. Aggregating images signals from all folders (name: {name})..")
            total_dict = aggregate_signals(self.log, total_stats[name], 'none') # Aggregate the signals of the current dataset in a single dict (format = 'metric': [list of values])
            save_aggregated_signals(self.log, self.images_folder, total_dict, name = dataset_name + ' ' + name) # Save the 'dict' on the root folder of the dataset.
        return None

    # TODO: Working for gather signal both from my dataset and others (e.g. CTC/DSB 2018 datasets)
    def __compute_signals(self, image_path, mask_path, perc_pixels, threshold):
        """ Load the image and mask from the given paths and compute the signals

        Args:
            image_path (str): Path to load the '*.tif' image
            mask_path (str): Path to load the '*.tif' mask
            perc_pixels (float): Percentage of the total backrounds to analyze singularly
            threshold (int): Number of pixels to filter out the cells.

        Returns:
            (dict): {'signals_1': (float), 'signals_2': (float)}
        """

        image = tiff.imread(image_path)
        
        if (len(image.shape) > 2):
            image = to_single_channel(image) # Convert the image if needed.
            self.log.debug(f".. image '{os.path.basename(image_path)}' converted to one channel keeping all objects ..")

        mask = tiff.imread(mask_path)
        signals_dict = {} # Init. the dict for the signals.

        boolean_mask = mask != 0 # Create a boolean matrix for the segmented objects.

        obj_pixels = len(mask[boolean_mask]) # Pixels belonging to segmented objects considered.
        background_pixels = image[~boolean_mask] # Take the pixel not belonging to the segmented objects.
        tot_pixels = mask.shape[0] * mask.shape[1] # Total pixels of the image.

        # NOTE: console visualization - debug
        print(f"\nWorking on {os.path.basename(image_path).split('.')[0]} image")
        print(f"> Rumore nel background: {np.std(background_pixels)}")
        print(f"> Numero di oggetti (incluso il background): {len(np.unique(mask))}")
        print(f"> Valori degli oggetti (incluso background): {np.unique(mask)}")

        # TODO: Make this adjusted to the current dataset.. my images have different number of pixels.
        mean_cells, std_cells, dim_cells, dim_cells_variations, cells_pixel = [], [], [], [], [] # Mean, std and number of pixels of the segmented cells (both total and )
        raw_obj_values, raw_obj_dims = np.unique(mask, return_counts = True) # Store the pixel value and the number of occurrence for each object
        
        # Filter out the segmented object using the thresholds - keeping the backgorund in every case.
        obj_values, obj_dims = [], []
        for val, count in zip(raw_obj_values, raw_obj_dims):
            
            if val == 0 or (count >= threshold[0] and count <= threshold[1]):
                obj_values.append(val), obj_dims.append(count)
   
        # NOTE: console visualization - debug
        print(f"> List format (thresholds {threshold}): 'pixels value' - 'number of pixel'")
        for value, count in zip(obj_values, obj_dims): # For now less than 7000 pixels is an EVs for sure (keeping into account that it depends on the image quality)
            print(f"   {value} - {count}")# The current obj. is a cell or background (background will occupy the FIRST position in the lists)
            current_mask = mask == value
            current_pixels = image[current_mask] # Take from the original image the pixels value corresponding to the current object mask
            
            # WARNING:  BUG on some dataset in the pixels value
            current_pixels[current_pixels > 255] = 255 # Cap the value of the pixels to the actual RGB limit channel values.
            cells_pixel.extend(current_pixels)
            mean_cells.append(np.mean(current_pixels))
            std_cells.append(np.std(current_pixels))
            dim_cells.append(len(current_pixels)/tot_pixels) # Get a ratio (this way it depends less on the resolution of the images)

        # To compute the image backround homogeinity
        print(f"> Background pixels shape: {background_pixels.shape}") # DEBUG console visualization
        
        step = floor(perc_pixels * len(background_pixels))
        background_patches = [] # List containing the avg. value of a patches of background pixels.
        for limit in range(step, len(background_pixels), step): # Take window of N pixels (N pixels given a certain percent of the total mask dim)
            
            current_patches = background_pixels[limit-step: limit] # Take one patches from the total pixels
            background_patches.append(np.mean(current_patches))
            print(f"   Background pixels patches - {len(current_patches)} - {np.mean(current_patches)}")
        
        # Visualization for DEBUG purpose - both image/mask saved inside the 'self.debug_folder'
        mask_name = os.path.basename(mask_path).split('.')[0]
        visualize_mask(mask, os.path.join(self.debug_folder, mask_name))
        image_name = os.path.basename(image_path).split('.')[0]
        visualize_image(image, os.path.join(self.debug_folder, image_name))
        
        signals_dict['cc'] = np.mean(mean_cells[1:]) # Cell color: mean of the cells values pixels (intensity).
        signals_dict['cv'] = np.mean(std_cells[1:]) # Cell variations: mean of the cells internal hetereogenity.
        signals_dict['ch'] = np.std(cells_pixel) # Cell hetereogenity: measure of the hetereogenity of the cells.
        signals_dict['cdr'] = np.mean(dim_cells[1:]) # Cell dimensions ratio: avg. of the number of pixels of the cells (in percentage over the total image).
        signals_dict['cdvr'] = np.std(dim_cells[1:]) # Cell dimensions variations ratio: std. of the number of pixels of the cells (in percentage over the total image).
        signals_dict['stn'] = obj_pixels/tot_pixels # Signal to noise ratio
        signals_dict['ccd'] = abs(signals_dict['cc'] - mean_cells[0])/255 # Cells Contrast Difference: Ratio in avg. pixel values between beackground and segmented cells over the maximum pixel value.
        signals_dict['bh'] = np.std(background_patches) # Background homogeinity: Measure the homogeinity of the different patches considering the std. of the average pixel values. 
        return signals_dict


class signalsVisualizator: # Object to plot signals of a single dataset (both aggregated/single mask folder): for now mantain this design - (To oeprate on multiple dataset just create a appropriate main script).
 
    def __init__(self, env, args, task='SEG'):
        """ Class to create a obj. that gather images signals from segmentation masks and plots the results"""

        self.log = env['logger'] # Extract the needed evn. variables
        self.args = args # TODO: Select just the needed args
        self.task = task # Folder for the ground truth mask signals loading

        os.makedirs(VISUALIZATION_FOLDER, exist_ok=True) # Set up a folder that will contains the final plots
        self.visualization_folder = VISUALIZATION_FOLDER # Path starting from the './' (current folder - root folder of the project)


    @staticmethod
    def dataset_signals_comparison(log, split_folder='training_data', target_folder=VISUALIZATION_FOLDER):
        """ Read all the 'aggregated_signals' from the different datasets folders and plots the comparison graph.

        Args:
            folders_sample (str): Folders to search for '*.json' to compare among different datasets (normally the training folder)

        Returns:
            None
        """
        os.makedirs(target_folder, exist_ok=True) # Set up a folder that will contains the final plots
        log.info(f"Comparison of different datasets signals (used the dataset folders in the split '{split_folder}')")
        
        dataset_folders = os.listdir(split_folder)
        dataset_folders = [s for s in dataset_folders if not s.startswith("__")]

        log.info(f"The datasets found are: {dataset_folders}")
        dataset_list = [] # Just append the name of the dataset that actual contains the data
        datasets_dict = defaultdict(list) # Dict in the format 'metric': [[one list for every dataset], [...], [...]]

        for dataset in dataset_folders:
            
            json_file = [s for s in os.listdir(os.path.join(split_folder, dataset)) if s.endswith(".json")] # Search for the 'aggregated_signals.json'
            if json_file: # Considering at least one 'aggregated_signals.json' for dataset.
                log.info(f".. extracting {dataset} data from {json_file}..")
                    
                for single_file in json_file:
                    dataset_list.append(single_file.split('.')[0])

                    f = open(os.path.join(split_folder, dataset, single_file)) # Open the current 'aggregated' data.
                    signals_dict = json.load(f) # The 'aggregated_signals.json' doesn't differentiate between '01_GT' and '02_GT'.

                    for metric, value in signals_dict.items(): # The 'value from the 'aggregated*.json' are lists of value
                        datasets_dict[metric].append(value)

            else:
                log.info(f".. {dataset} doesn't contains any data ..")
                        
        log.debug(f"Total data considered are {dataset_list}")

        log.info(f"The graphs will be saved in '{target_folder}'")
        signalsVisualizator.__box_plots(log, datasets_dict, dataset_list, target_folder)
        signalsVisualizator.__line_plots(log, datasets_dict, dataset_list, target_folder)
        return None
    
    @staticmethod
    def __box_plots(log, datasets_dict, dataset_list, target_folder):

        for key, lists in datasets_dict.items():
            # Creare i boxplots per ogni lista nella chiave corrente
            plt.figure(figsize=(14, 10))

            plt.boxplot(lists)

            # Impostare le etichette sull'asse x
            plt.xticks(np.arange(1, len(dataset_list) + 1), dataset_list, rotation=45) # set the x ticks

            plt.ylabel(f'{key}')
            plt.title(f'Dataset comparison')

            plt.savefig(os.path.join(target_folder, f"{key}_boxplot"))
            plt.close()
        return None

    
    @staticmethod
    def __line_plots(log, datasets_dict, dataset_list, target_folder):
        
        for key, lists in datasets_dict.items():

            plt.figure(figsize=(14, 10))

            for idx, values in enumerate(lists): # Plot the curve for every list for that metric
                
                plt.plot(values, '-.', label = dataset_list[idx])
                
                plt.legend(fontsize="8")
                plt.xlabel("Time frame")
                plt.ylabel(key)

                plt.savefig(os.path.join(target_folder, f"{key}_lineplot"))
            plt.close() # Close the picture of this metric