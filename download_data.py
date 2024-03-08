"""download_data.py

This file contains the function to download and set up the train/test data 
and the evaluation software.
"""

import os
import requests
import zipfile
import tifffile as tiff

from utils import *
from img_processing.imageUtils import *
from parser import get_parser, get_processed_args


def main():
    """ Function to call the download data utils.
    """
    set_environment_paths_and_folders()
    log = create_logging() # Set up 'logger' object 

    args = get_parser() # Set up dict arguments
    args = get_processed_args(args)

    env = {} # TODO: Load this from a '.json' file
    env['logger'] = log # Move the object through 'env' dict

    log.info(f"args: {args}") # Print overall args 
    log.debug(f"env: {env}")

    log.info(f"--- Downloading data and software ---")
    download_datasets(log, args) # Call the 'util' function.
    check_and_download_evaluation_software(log, args.evaluation_software_path)
    log.info("Downloads ended correctly :)")

if __name__ == "__main__":
    main()