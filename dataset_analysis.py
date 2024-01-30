"""dataset_analysis.py

This is the main executable file for gather images properties and compare 
signals among the images for dataset benchmarking.
"""
from utils import create_logging
from parser import get_parser, get_processed_args
from img_processing.main_img import * # Import gather and visualizator of images property
from download_data import download_datasets


def main():
    """ Main function to call in order to run all the project classes and 
        functions about images download and properties gathering.
    """
    log = create_logging() # Set up 'logger' object 

    args = get_parser() # Set up dict arguments
    args = get_processed_args(args)

    env = {} # TODO: Load this from a '.json' file
    env['logger'] = log # Move the object through 'env' dict

    log.info(f"Args: {args}") # Print overall args.
    log.debug(f"Env varibles: {env}")

    log.info(f"--- Computing image characteristics ---")

    # Process single folders signals and aggregate for the current dataset chosen by args.
    for dataset in args.dataset:
        processor = images_processor(env, args, dataset)
        processor.collect_signals()

    if args.compare_signals:
        visualizator = signalsVisualizator(env, args)
        #visualizator.visualize_signals() # WORK IN PROGRESS: Compute single dataset signals.
        signalsVisualizator.dataset_signals_comparison(log) # Compare single signals from different datasets (when available).


if __name__ == "__main__":
    main() # For now just donwload, compute and visualize signals


