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
        log.info(f"Creation of the training dataset using {args.pre_processing_pipeline} pipeline")
        create_ctc_training_sets(log, path_data=path_data, mode=args.mode, cell_type=args.dataset, split=args.split, min_a_images=args.min_a_images)
    else:
        raise ValueError("This argument support just 'kit-ge' as pre-processing pipeline")


    # WORK IN PROGRESS
    log.info(">>> Training script ended correctly <<<")


# Implementing 'kit-ge' training method - consider to make unique for every chosen pipeline/make modular later.
def kit_ge_dataset_creation(log, models, path_models, train_sets, path_data, device, scale_factor, args):
    pass


if __name__ == "__main__":
    main()