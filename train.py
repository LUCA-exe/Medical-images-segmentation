"""train.py

This is the train main function that call creation of training set of a dataset
and train the models.
"""
import os
from os.path import join, exists
from collections import defaultdict
from utils import create_logging, set_device, EvalArgs
from parser import get_parser, get_processed_args
from inference.inference import inference_2d # Main inference loop
from net_utils.metrics import count_det_errors, ctc_metrics, save_metrics

from training.create_training_sets import create_ctc_training_sets


def main():
    """ Main function to set up paths, datasets and training pipelines
    """
    log = create_logging() # Set up 'logger' object 

    args = get_parser() # Set up dict arguments
    args = get_processed_args(args)

    env = {} # TODO: Load this from a '.json' file environment parameters
    env['logger'] = log # Move the object through 'env' dict

    log.info(f"Args: {args}") # Print overall args 
    log.debug(f"Env varibles: {env}")
    device = set_device() # Set device: cpu or single-gpu usage
    log.info(f">>>   Training: pre-processing {args.pre_processing_pipeline} model {args.model_pipeline} <<<")

    # Load paths
    path_data = join(args.train_images_path, args.dataset)
    path_models = args.models_folder # Train all models found here.

    # TODO: Move this into 'utils' file (called both in 'val' and 'train')
    if not exists(path_data):
        log.info(f"Warning: the '{path_data}' provided is not existent! Interrupting the program...")
        raise ValueError("The '{path_data}' provided is not existent")
    else:
        trainset_name = args.dataset # 'atgs.dataset' used as cell type

    # Pre-processing pipeline - implement more pipeline from other papers here ..
    if args.pre_processing_pipeline == 'kit-ge':
        create_ctc_training_sets(path_data=path_data, mode=args.mode, cell_type_list=args.cell_type, split=args.split)
    else:
        raise ValueError("This argument support just 'kit-ge' as pre-processing pipeline")

    # Set-up model name
    #WORK IN PROGRESS
        
    # Load all models in the chosen folder
    models = [model for model in os.listdir(args.models_folder) if model.endswith('.pth')]

    # Load the paths in the log files
    log.debug(f"Dataset folder to evaluate: {path_data}")
    log.debug(f"Folder used to fetch models: {path_models}")
    log.debug(f"Folder used to save models performances/files: {path_data}") # Saving the results on the folder of the evaluated dataset
    if path_ctc_metric != 'none': # In case of 'none' (str) use custom metrics on the script
        log.debug(f"Evaluation software folder: {path_ctc_metric}")

    scores = [] # Temporary list to keep the evaluation results
    train_sets = args.subset # List of subfolder to eval: already parser from args
    
    # NOTE: For now it is implemented evaluation for one dataset
    if args.post_processing_pipeline == 'kit-ge': # Call inference from the KIT-GE-(2) model's method
        kit_ge_inference_loop(log, models, path_models, train_sets, path_data, device, args.scale, args)
    
    else: # Call other inference loop ..
        raise NotImplementedError(f"Other inference options not implemented yet ..")
    
    log.info(">>> Evaluation script ended correctly <<<")


# Implementing kit-ge pre-processing method.
def kit_ge_dataset_creation(log, models, path_models, train_sets, path_data, device, scale_factor, args):
    pass


if __name__ == "__main__":
    main()