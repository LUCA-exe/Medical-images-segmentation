"""eval.py

This is the evaluation main function that call inference and the metrics computation.
"""
from utils import create_logging, download_images
from parser import get_parser, get_processed_args
from download_data import download_datasets




def main():
    """ Main function to call in order to run all the project classes and functions about image download and properties
    """
    log = create_logging() # Set up 'logger' object 

    args = get_parser() # Set up dict arguments
    args = get_processed_args(args)

    env = {} # TODO: Load this from a '.json' file
    env['logger'] = log # Move the object through 'env' dict

    log.info(f"Args: {args}") # Print overall args 
    log.debug(f"Env varibles: {env}")

    # Load paths
    

    


if __name__ == "__main__":
    main()