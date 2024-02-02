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

    def __init__(self, env, args, dataset, split_signals=False, task='SEG'):
        """Class to create an obj. that gather images signals from segmentation masks"""

        self.log = env['logger'] # Extract the needed env. variables - log
        self.args = args
        self.images_folder = os.path.join(args.train_images_path, dataset) # Path in which the json will be saved
        self.task = task # Final folder for the ground truth mask
        self.thr = args.cell_dim
        self.considered_images = args.max_images # Set a limit to the number of masks to use
        self.split_signals = split_signals

        if self.split_signals: # In case of 'True', set the name of the '*.json' signals file
            self.names = ['EVs', 'Cells']
        else:
            self.names = [''] # In case of a single 'signals' file for GT folder.

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
        self.log.info(f"Folders inside the dataset: {folders}")

        masks_folders = [s for s in folders if s.endswith("GT")]
        self.log.debug(f"Folders with the masks: {masks_folders}")
        
        total_stats = [] # List of dicts containing all the computed signals - It is not a 'dict' cause we can lose the 'name' of the image for the final aggregation.
        if len(self.names) > 1:
            self.log.info(f"Computing the characteristics of multiple cells in the images per folder")

        for folder in masks_folders: # Main loop: for every folder gather the data.

            # NOTE: Added the split of signals based on cells dimension - if requested by the '--split_signals' arg.
            for name in self.names:
                
                current_images_path = os.path.join(self.images_folder, folder.split('_')[0]) # Fetch the original images from this folder
                current_path = os.path.join(self.images_folder, folder, self.task) # Compose the masks folder

                files_name = [s for s in os.listdir(current_path) if s.startswith("man")] # Get the file names of the masks
                files_name.sort()
                self.log.debug(f"Currently working in '{current_path}': files are {files_name}")

                create_signals_file(self.log, current_path) # Prepare the '*.json' signals file to store the data
                stats = {} # Dict contatining 'id' and {'signal' : value} of masks of a single folder

                if len(files_name) > self.considered_images: files_name = files_name[:self.considered_images] # Consider just a limited number of masks for current folders
                
                for file_name in files_name:
                    current_mask_path = os.path.join(current_path, file_name)
                    image_name = fetch_image_path(current_mask_path, current_images_path) # For evey mask path, fetch the proper image path (images masks less than total of the images)
                    current_image_path = os.path.join(current_images_path, image_name)

                    # Ready to compute the signals for the coupled mask - image
                    self.log.debug(f".. working on {image_name} - {file_name} ..")
                    stats[image_name] = self.__compute_signals(current_image_path, current_mask_path, perc_pixels) # Save the signals for every original image name

                    total_stats.append(deepcopy(stats[image_name])) # Store the image signals for the future aggregation of this dataset (copy, not the reference)

                # Finished to gather data from the current folder - update the existing '*.json' (on the current folder)
                update_signals_file(self.log, current_path, stats)

            self.log.info(f".. Aggregating images signals from all folders ..")
            total_dict = aggregate_signals(self.log, total_stats, 'none') # Aggregate the signals of the current dataset in a single dict (format = 'metric': [list of values])
            save_aggregated_signals(self.log, self.images_folder, total_dict, name=) # Save the 'dict' on the root folder of the dataset.
        return None

    # TODO: Working for gather signal both from my dataset and others (e.g. CTC/DSB 2018 datasets)
    def __compute_signals(self, image_path, mask_path, perc_pixels):
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
            self.log.debug(f".. image '{os.path.basename(image_path)}' converted to one channel keeping all objects ..")

        mask = tiff.imread(mask_path)
        signals_dict = {} # Init. the dict for the signals

        boolean_mask = mask != 0 # Create a boolean matrix for the segmented objects

        obj_pixels = len(mask[boolean_mask]) # Pixels belonging to segmented objects considered
        background_pixels = image[~boolean_mask] # Take the pixel not belonging to the segmented objects
        tot_pixels = mask.shape[0] * mask.shape[1] # Total pixels of the image

        # NOTE: console visualization - debug
        print(f"\nWorking on {os.path.basename(image_path).split('.')[0]} image")
        print(f"> Rumore nel background: {np.std(background_pixels)}")
        print(f"> Numero di oggetti (incluso il background): {len(np.unique(mask))}")
        print(f"> Valori degli oggetti (incluso background): {np.unique(mask)}")

        # TODO: Make this adjusted to the current dataset.. my images have different number of pixels
        # For now less than 7000 pixels is an EVs for sure (keeping into account that it depends on the image resolution)
        mean_cells, std_cells, dim_cells, dim_cells_variations, cells_pixel = [], [], [], [], [] # Mean, std and number of pixels of the segmented cells (both total and )
        obj_values = np.unique(mask) # Store the pixel value for each object
        obj_dims = [np.count_nonzero(mask==value) for value in obj_values] # Numbers of pixels occupied for every objects (in order of total pixels)
        
        # DEBUG console visualization
        print(f"> List format: 'pixels value' - 'number of pixel'")
        for value, count in zip(obj_values, obj_dims): # For now less than 7000 pixels is an EVs for sure (keeping into account that it depends on the image quality)
            
            print(f"   {value} - {count}")
            if count > self.thr: # The current obj. is a cell or background (background will occupy the FIRST position in the lists)
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
        
        signals_dict['cc'] = np.mean(mean_cells[1:]) # Cell color: mean of the cells values pixels.
        signals_dict['cv'] = np.mean(std_cells[1:]) # Cell variations: mean of the cells internal hetereogenity.
        #signals_dict['ch'] = np.std(cells_pixel) # Cell hetereogenity: measure of the hetereogenity of the signals among the cells in the frame.
        signals_dict['ch'] = np.std(mean_cells[1:]) # Cell hetereogenity: measure of the hetereogenity of the signals among the cells in the frame.
        signals_dict['cdr'] = np.mean(dim_cells[1:]) # Cell dimensions ratio: avg. of the number of pixels of the cells (in percentage over the total image)
        signals_dict['cdvr'] = np.std(dim_cells[1:]) # Cell dimensions variations ratio: std. of the number of pixels of the cells (in percentage over the total image)
        signals_dict['stn'] = obj_pixels/tot_pixels # Signal to noise ratio
        signals_dict['bn'] = std_cells[0] # Background noise - computed during the stats over the segmented obj.

        # TODO: Contrast ratio of the segmented EVs and Cells - for now just cells in order to compute the metric for the other dataset
        signals_dict['ccd'] = abs(np.mean(mean_cells[1:]) - mean_cells[0])/255 # Cells Contrast Difference: Ratio in avg. pixel values between beackground and segmented cells over the maximum pixel value
        signals_dict['bh'] = abs(max(background_patches) - min(background_patches)) # Background homogeinity: Measure the homogeinity of the different avg. pixels values of the background patches 
        
        return signals_dict


class signalsVisualizator: # Object to plot signals of a single dataset (both aggregated/single mask folder): for now mantain this design - (To oeprate on multiple dataset just create a appropriate main script).
 
    def __init__(self, env, args, task='SEG'):
        """ Class to create a obj. that gather images signals from segmentation masks and plots the results"""

        self.log = env['logger'] # Extract the needed evn. variables
        self.args = args # TODO: Select just the needed args
        #self.images_folder = os.path.join(args.train_images_path, args.dataset) # Path to load the different '*.json' files
        self.images_folder = None
        self.task = task # Folder for the ground truth mask signals loading

        os.makedirs(VISUALIZATION_FOLDER, exist_ok=True) # Set up a folder that will contains the final plots
        self.visualization_folder = VISUALIZATION_FOLDER # Path starting from the './' (current folder - root folder of the project)

        self.log.info(f"Obj. to plot the computed signals instantiated: working on '{self.images_folder}'")


    # TODO: Finish th eplotting function
    def visualize_signals(self): # Save a plot for every '*.json' that can find

        # DEBUG console
        print(f"Files in the {self.images_folder} are {os.listdir(self.images_folder)}")

        files = os.listdir(self.images_folder) # List files contained in the current dataset folder
        # Search for '.json' extension (the aggregated signals of the dataset)
        aggr_json = [s for s in files if s.endswith(".json")]
        
        if aggr_json: # If the obj. is True then is not empty
            self.log.info(f"Aggregated file {aggr_json} found!")
            aggr_json = aggr_json[0] # Select just the value from the list to load the file
            f = open(os.path.join(self.images_folder, aggr_json)) # Open the aggregated file
            aggr_data = json.load(f)
            # TODO: Select how to visualize/compare the aggr_data with the other datasets
        else:
            self.log.info(f"Aggregated file {aggr_json} NOT found.. searching for the specific folders signals ..")
        
        mask_folders = [s for s in files if s.endswith("_GT")] # Take all the mask folders to extract the single '*.json' for each one
        for mask_folder in mask_folders:
            
            current_files = os.listdir(os.path.join(self.images_folder, mask_folder, self.task))
            print(f"Current files for the '{mask_folder}' are: {current_files}")
            
            data_json  = [s for s in current_files if s.endswith(".json") and s.startswith('dataset')]
            if data_json: # Check for the signal file of the current folder
                data_json = data_json[0]
                current_data_json = os.path.join(self.images_folder, mask_folder, self.task, data_json)
                f = open(current_data_json) # Open the aggregated file
                data = json.load(f) # Signals of the current folder
                
                data_list = [] # List of images signals of the current folder
                for key in data.keys(): # Create a list of dict (one dict for every image)
                    data_list.append(deepcopy(data[key]))
                
                data_list = aggregate_signals(self.log, data_list, 'none') # Get a dict with for every metric key the list of values for this current folder (lose the image name key)
                self.__plot_signals(data_list, self.args.dataset + '_' + mask_folder + '_', self.visualization_folder)
            
        return None

    # TODO: Finish the plotting function.
    def __plot_signals(self, data, file_name, path): # Plot the dict metrics of a dataset's single folder (every metric contains a list of value)
        """ Read the list of dict and plot a graph for every metrics contained

        Args:
            data (list): List of dict; every key contains a list of values (one for every image)
            file_name (str): It contains the name of the file (format: 'dataset_imagesfolder')

        Returns:
            None
        """
        # DEBUG
        print(f"file name : {file_name}")
        print(f"file path : {path}")

        for metric, values in data.items():
            # DEBUG
            print(f"Metric :{metric}   -   values: {values}")
            current_file_path = os.path.join(path, file_name + metric)
            print(f"saving {current_file_path}")
            
            # Plot the single metric
            plt.plot(range(len(values)), values, 'r')
            plt.xlabel("values")
            plt.ylabel(metric)
            plt.savefig(current_file_path)
            plt.close()

        return None


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
                for file in json_file
                    
                    log.info(f".. extracting {dataset} data ..")
                    dataset_list.append(dataset)

                    f = open(os.path.join(split_folder, dataset, json_file[0])) # Open the current 'aggregated' data.
                    signals_dict = json.load(f) # The 'aggregated_signals.json' doesn't differentiate between '01_GT' and '02_GT'.

                    for metric, value in signals_dict.items(): # The 'value from the 'aggregated*.json' are lists of value
                        datasets_dict[metric].append(value)

            else:
                log.info(f".. {dataset} doesn't contains any data ..")
                        
        log.debug(f"Total dataset considered are : {dataset_list}")

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

    




































