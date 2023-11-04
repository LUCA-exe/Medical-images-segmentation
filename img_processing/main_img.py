"""main_img.py

This is the main executable file for running the processing functions for the images/masks.
"""

from copy import deepcopy
import json
from math import floor
import os
import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff
import cv2
from collections import defaultdict
from img_processing.imageUtils import * # Remember to write the path the the 'importer' of this file is calling

class images_processor:

    def __init__(self, env, args, task='SEG'):
        """Class to create a obj. that gather images signals from segmentation masks"""

        self.log = env['logger'] # Extract the needed evn. variables
        self.args = args # TODO: Select just the needed args
        self.images_folder = os.path.join(args.train_images_path, args.dataset) # Path in which the json will be saved
        self.task = task # Final folder for the ground truth mask
        self.thr = args.cell_dim

        os.makedirs('tmp', exist_ok=True) # Set up a './tmp' folder for debugging images
        self.debug_folder = 'tmp'

        self.log.info(f"'Image processor' object instantiated: working on '{self.images_folder}'")
        
        
    def set_dataset(self, new_images_path, new_dataset): # Change the target repository
        self.images_folder = os.path.join(new_images_path, new_dataset)
        self.log.info(f"Dataset considered by the processor object changed correctly to {new_dataset}!" )
        return None

    # TODO: Verifiy that 'perc_pixels' should be a parameters callable from the main and NOT a paramereter on the program
    def collect_signals(self, perc_pixels = 0.2):
        """ Load the images from a folder and compute the signals

        Args:
            perc_pixels (float): Percent value used for a metric computation in the images (in particular for '')

        Returns:
            None 
        """
        # This function will process images; cleaning the './tmp' folder
        if os.listdir(self.debug_folder): 
            # 'Debug dir' has some files: delete all
            self.log.debug(f"Folder '{self.debug_folder}' contains files, It will be cleaned")
            for file_name in os.listdir(self.debug_folder):
                os.remove(os.path.join(self.debug_folder, file_name))
             
        # Start collecting the folders path for the current dataset
        folders = os.listdir(self.images_folder)
        self.log.info(f"Folders inside the dataset: {folders}")

        masks_folders = [s for s in folders if s.endswith("GT")]
        self.log.debug(f"Folders with the masks: {masks_folders}")
        
        total_stats = [] # List of dicts containing all the computed signals - It is not a 'dict' cause we can lose the 'name' of the image for the final aggregation

        for folder in masks_folders: # Main loop: for every folder gather the data

            current_images_path = os.path.join(self.images_folder, folder.split('_')[0]) # Fetch the original images from this folder
            current_path = os.path.join(self.images_folder, folder, self.task) # Compose the masks folder

            files_name = [s for s in os.listdir(current_path) if s.startswith("man")] # Get the file names of the masks
            files_name.sort()
            self.log.debug(f"Currently working in '{current_path}': files are {files_name}")

            create_signals_file(self.log, current_path) # Prepare the '*.json' signals file to store the data - 
            stats = {} # Dict contatining 'id' and {'signal' : value} of masks of a single folder

            for file_name in files_name:
                current_mask_path = os.path.join(current_path, file_name)
                # For evey mask path, fetch the proper image path
                image_name = fetch_image_path(current_mask_path, current_images_path)
                current_image_path = os.path.join(current_images_path, image_name)

                # Ready to compute the signals for the coupled mask - image
                stats[image_name] = self.__compute_signals(current_image_path, current_mask_path, perc_pixels) # Save the signals for every original image name

                total_stats.append(deepcopy(stats[image_name])) # Store the image signals for the future aggregation of this dataset

            # Finished to gather data from the current folder - update the existing '*.json' (on the current folder)
            update_signals_file(self.log, current_path, stats)

        total_dict = aggregate_signals(self.log, total_stats) # Aggregate the signals of the current dataset in a single dict 
        save_aggregated_signals(self.log, self.images_folder, total_dict) # Save the 'dict' on the root folder of the dataset

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
            self.log.debug(f".. image '{os.path.basename(image_path)}' converted to one channel ..")

        mask = tiff.imread(mask_path)
        signals_dict = {} # Init. the dict for the signals

        boolean_mask = mask != 0 # Create a boolean matrix for the segmented objects

        obj_pixels = len(mask[boolean_mask]) # Pixels belonging to segmented objects considered
        tot_pixels = mask.shape[0] * mask.shape[1] # Total pixels of the image

        # DEBUG console visualization
        print(f"Working on {os.path.basename(image_path).split('.')[0]} image")
        background_pixels = image[~boolean_mask] # Take the pixel not belonging to the segmented objects
        print(f"> Rumore nel background: {np.std(background_pixels)}")
        print(f"> Numero di oggetti (incluso il background): {len(np.unique(mask))}")
        print(f"> Valori degli oggetti (incluso background): {np.unique(mask)}")

        # TODO: Make this adjusted to the current dataset.. my images have different number of pixels
        #thr = 7000 # For now less than 7000 pixels is an EVs for sure (keeping into account that it depends on the image quality)
        mean_cells, std_cells = [], [] # Mean and std inside the segmented cells
        obj_values = np.unique(mask) # Store the pixel value for each object
        obj_dims = [np.count_nonzero(mask==value) for value in obj_values] # Numbers of pixels occupied for every objects (in order of total pixels)
        
        for value, count in zip(obj_values, obj_dims): # For now less than 7000 pixels is an EVs for sure (keeping into account that it depends on the image quality)
            print(f"> List of 'pixels value' - 'number of pixel'")
            print(f"   {value} - {count}")

            if count > self.thr: # The current obj. is a cell or background (background will occupy the FIRST position in the lists)
                current_mask = mask == value
                current_pixels = image[current_mask] # Take from the original image the pixels value corresponding to the current object mask
                mean_cells.append(np.mean(current_pixels)) 
                std_cells.append(np.std(current_pixels))

        # To compute the image backround homogeinity
        print(f"> Background pixels shape: {background_pixels.shape}")
        #for patch in (background_pixels):
        
        step = floor(perc_pixels * len(background_pixels))
        background_patches = [] # List containing the avg. value of a patches of background pixels
        for limit in range(step, len(background_pixels), step): # Take window of N pixels (N pixels given a certain percent of the total mask dim)
            
            current_patches = background_pixels[limit-step: limit] # Take one patches from the total pixels
            background_patches.append(np.mean(current_patches))
            print(f"   Background pixels patches - {len(current_patches)} - {np.mean(current_patches)}")
        
        #mask = np.ma.masked_array(mask, mask==0) # TODO: Check if the mask has to be modified already here - It should just be a conversion of array 0 values 

        # Visualization for debug purpose - both image/mask saved inside the 'self.debug_folder'
        mask_name = os.path.basename(mask_path).split('.')[0]
        visualize_mask(mask, os.path.join(self.debug_folder, mask_name))
        image_name = os.path.basename(image_path).split('.')[0]
        visualize_image(image, os.path.join(self.debug_folder, image_name))
        
        signals_dict['cc'] = np.mean(mean_cells[1:]) # Cell color: mean of the cells pixels
        signals_dict['cv'] = np.mean(std_cells[1:]) # Cell variations: mean of the cells pixels std
        signals_dict['stn'] = obj_pixels/tot_pixels # Signal to noise ratio
        signals_dict['bn'] = std_cells[0] # Background noise - computed during the stats over the segmented obj.

        # TODO: Contrast ratio of the segmented EVs and Cells - for now just cells in order to compute the metric for the other dataset
        signals_dict['crc'] = abs(np.mean(mean_cells[1:]) - mean_cells[0]) # Cells Contrast Ratio: Absoluth difference in avg. pixel values between beackground and segmented cells
        
        # TODO: Debug the computing loop for the patches gathering
        signals_dict['bh'] = abs(max(background_patches)-min(background_patches)) # Background homogeinity: Measure the homogeinity of the different avg. pixels values of the background patches 
        
        # TODO: Cells hetereogenity along the time lapse, aggregated measure: variations of the avg values of the cells along the different frames

        return signals_dict

    # TODO: Remove from here - moved to utils
    def __aggregate_signals(self, signals_list, method='mean'):
        """ Create a dict that contains aggregated signals gathered for every image of a dataset.

        Args:
            signals_list (obj): List of dict; every dict contains the signals of a single image

        Returns:
            (dict): Return a unique dict of aggregated signals
        """

        # TODO: Upgrade and add functionalities: for now just compute the mean for every aggregated signals 
        total_dict = defaultdict(list)
        
        for single_dict in signals_list: # for every image dict

            for key, value in single_dict.items(): # Append the value to the 'total_dict'
                total_dict[key].append(value)

        for k in total_dict.keys(): # Just aggregate with the chosen method
            total_dict[k] = np.mean(total_dict[k]) # Even if the defaultdict is set to list It works just with void key.

        self.log.debug(f"Signals of the current dataset aggregated correctly!")
        return total_dict


# TODO: Follow the standard format for name object classes
class signalsVisualizator: # Object to plot signals of a single dataset (both aggregated/single mask folder): for now mantain this design - (To oeprate on multiple dataset just create a appropriate main script).
 
    def __init__(self, env, args, task='SEG'):
        """ Class to create a obj. that gather images signals from segmentation masks"""

        self.log = env['logger'] # Extract the needed evn. variables
        self.args = args # TODO: Select just the needed args
        self.images_folder = os.path.join(args.train_images_path, args.dataset) # Path to load the different '*.json' files
        self.task = task # Folder for the ground truth mask signals loading

        os.makedirs('visualization_results', exist_ok=True) # Set up a folder that will contains the final plots
        self.visualization_folder = 'visualization_results' # Path starting from the './' (current folder - root folder of the project)

        self.log.info(f"Obj. to plot the computed signals instantiated: working on '{self.images_folder}'")

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

    # NOTE: WORK IN PROGRESS! (To debug the actual values..)
    @staticmethod
    def plot_signals_comparison(log, split_folder='training_data', folders_sample = '01_GT', task='SEG'):
        """ Read the list of dict (one for each datasets) and plot a graph for every metrics contained

        Args:
            folders_sample (str): Folders to search for '*.json' to compare among different datasets

        Returns:
            None
        """

        os.makedirs('visualization_results', exist_ok=True) # Set up a folder that will contains the final plots
        log.info(f"Comparison of different datasets signals (used the folders '{folders_sample}' in the split '{split_folder}')")
        # Comparare le metriche divise per folder divise per dataset - salvare file su visualization results
        
        # For every dataset, take the signals computed in the '01_GT' folder
        dataset_folders = os.listdir(split_folder)
        dataset_folders = [s for s in dataset_folders if not s.startswith("__")]
        print(dataset_folders) 
        
        # List of dict: every dict contains the metric and values of the images in a 'folders_sample' of the current dataset
        comparison_list = [] 
        names_list = [] # Name of the dataset (for the label in the final plots)

        for folder in dataset_folders: # Enter in the single mask folder
            curr_json_path = os.path.join(split_folder, folder, folders_sample, task)
            print(curr_json_path)
            log.info(f".. searching signals for '{curr_json_path}'")
            
            # TODO: Make this an args/const/params in a config file
            json_file = 'dataset_signals.json'

            if json_file in os.listdir(curr_json_path):
                f = open(os.path.join(curr_json_path, json_file)) # Open the single folder 'signals'
                signals_dict = json.load(f)
                log.info(f"File found succesfully!")

                # DEBUG
                print(f"data : {signals_dict}")

                # TODO: Make this an utils function.. used often
                data_list = [] # List of images signals of the current folder
                for key in signals_dict.keys(): # Create a list of dict (one dict for every image)
                    data_list.append(deepcopy(signals_dict[key]))

                # Store the values for every metric of the current folder
                comparison_list.append(aggregate_signals(log, data_list, 'none')) # Get list of values for every metrics starting from the signals dict

            else:
                # Just explore other folders
                log.info(f"File not found .. still to compute!")
        
        # DEBUG
        print(f"data (converted) : {comparison_list}")
        print(f"Number of dataset read: {len(comparison_list)}")

        # TODO: Move this to another function (utils - plot a sequence of dict)
        if comparison_list:
            metrics = list(comparison_list[0].keys()) # Cast to list from 'dict_keys' obj.
            print(f"Keys to print: {metrics}")
            
            # Plot for every metrics all the values for every dataset (of the sample folder chosen)
            for key in metrics:
                
                current_values = [] # List to collect the values for every dataset for a metrics
                for single_dict in comparison_list:
                    current_values.append(single_dict[key])

                # DEBUG
                print(f"For the key '{key}' there are {len(current_values)} array of values to plot!")

                # Open, plot the array with the proper name and save the fig. for every metric
                for idx, values in enumerate(current_values):
                    plt.plot(values, 'o', label = dataset_folders[idx])
                
                plt.legend(fontsize="8")
                plt.xlabel("Time frame")
                plt.ylabel(key)

                plt.savefig(os.path.join('visualization_results', key))
                plt.close() # Close the picture of this metric


        else:
            # No signals json is found ..
            log.info(f"There are not signals computed in the datasets!")
            

        return None



































