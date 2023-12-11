"""eval.py

This is the evaluation main function that call inference and the metrics computation.
"""
import os
from os.path import join, exists
from utils import create_logging, download_images
from parser import get_parser, get_processed_args
from download_data import download_datasets


def main():
    """ Main function to set up paths, load model and evaluate the inferred images.
    """
    log = create_logging() # Set up 'logger' object 

    args = get_parser() # Set up dict arguments
    args = get_processed_args(args)

    env = {} # TODO: Load this from a '.json' file environment parameters
    env['logger'] = log # Move the object through 'env' dict

    log.info(f"Args: {args}") # Print overall args 
    log.debug(f"Env varibles: {env}")
    log.debug(f"-----------------------------------------------\nEvaluation")

    # Load paths
    path_data = join(args.train_images_path, args.dataset)
    path_models = args.models_folder # Eval all models found here.
    path_best_models = args.save_model # Save best model files/metrics here
    path_ctc_metric = args.evaluation_software

    if not exists(path_data):
        log.debug(f"Warning: the '{path_data}' provided is not existent! Interrupting the program...")
        raise ValueError("The '{path_data}' provided is not existent")

    models = [model for model in os.listdir(args.models_folder) if model.endswith('.pth')]

    # Load the paths in the log files
    log.debug(f"Dataset folder to evaluate: {path_data}")
    log.debug(f"Folder used to fetch models: {path_models}")
    log.debug(f"Folder used to save models performances/files: {path_best_models}")
    if path_ctc_metric != 'none': # In case of 'none' (str) use custom metrics on the script
        log.debug(f"Evaluation software folder: {path_ctc_metric}")

    # Providing a gridsearch for finetunable parameters - kept from the original repositories
    if not isinstance(args.th_seed, list):
        args.th_seed = [args.th_seed]
    if not isinstance(args.th_cell, list):
        args.th_cell = [args.th_cell]
    

    

    


if __name__ == "__main__":
    main()