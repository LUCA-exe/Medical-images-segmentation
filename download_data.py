"""download_data.py

This file contains the function to download and set up the train/test data.

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
    log = create_logging() # Set up 'logger' object 

    args = get_parser() # Set up dict arguments
    args = get_processed_args(args)

    env = {} # TODO: Load this from a '.json' file
    env['logger'] = log # Move the object through 'env' dict

    log.info(f"args: {args}") # Print overall args 
    log.debug(f"env: {env}")

    download_datasets(log, args) # Call the 'util' function

    # NOTE: Add the evalutation sofwtare donwload id resuested and not already present.
    log.info("Downloads ended correctly :)")

if __name__ == "__main__":
    main()