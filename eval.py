"""eval.py

This is the evaluation main function that call inference and the metrics computation.
"""
from utils import create_logging, download_images
from parser import get_parser, get_processed_args
from img_processing.main_img import * # Import gather and visualizator of images property
from download_data import download_datasets


def main():
    """ Main function to call in order to run all the project classes and functions about image download and properties
    """
    log = create_logging() # Set up 'logger' object 

    args = get_parser() # Set up dict arguments
    args = get_processed_args(args)

    env = {} # TODO: Load this from a '.json' file
    env['logger'] = log # Move the object through 'env' dict

    log.info(f"args: {args}") # Print overall args 
    log.debug(f"env: {env}")

    if args.download: # Check if is is requested the donwloading of datasets
        download_datasets(log, args)

    if args.compute_signals:
        # Process single folders signals and aggregate for the current dataset chosen by args
        processor = images_processor(env, args)
        processor.collect_signals()

    if args.compare_signals:
        visualizator = signalsVisualizator(env, args)
        #visualizator.visualize_signals() # WORK IN PROGRESS: Compute single dataset signals
        signalsVisualizator.dataset_signals_comparison(log) # Compare single signals from different datasets


if __name__ == "__main__":
    main() # For now just donwload, compute and visualize signals


