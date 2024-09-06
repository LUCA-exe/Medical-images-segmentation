"""train.py

This is the train main function that call creation of training set of a dataset.
and train the models.
"""

import os
from pathlib import Path
from os.path import join, exists
from collections import defaultdict
from utils import create_logging, set_device, set_environment_paths_and_folders, check_path, train_factory
from parser import get_parser, get_processed_args
from net_utils.utils import unique_path, write_train_info, create_model_architecture
from net_utils import unets
from training.create_training_sets import create_ctc_training_sets, get_file
from training.training import train, train_auto, get_max_epochs, get_weights
from training.mytransforms import augmentors
from training.autoencoder_dataset import AutoEncoderDataset
from training.cell_segmentation_dataset import CellSegDataset


def set_up_args_and_folder():
    # Parse initial arguments and set up environment variables/folder.

    set_environment_paths_and_folders() # Read .env file and set-up the temporary folders

    log = create_logging() # Set up 'logger' object and set up the current run folders
    
    args = get_parser() # Set up dict arguments
    args = get_processed_args(args)
    return args, log


def set_up_training_set(log, args, path_data, cell_type):
    # Pre-processing pipeline among the ones implemented to create the training set

    if args.pre_processing_pipeline == 'kit-ge':
        log.info(f"Creation of the training dataset using {args.pre_processing_pipeline} pipeline for {args.crop_size} crops")
        
        for crop_size in args.crop_size: # If you want to create more dataset
            create_ctc_training_sets(log, path_data=path_data, mode=args.mode, cell_type=cell_type, split=args.split, min_a_images=args.min_a_images, crop_size = crop_size)
    else:
        raise ValueError("This argument support just 'kit-ge' as pre-processing pipeline")


def get_training_args_class(log, args, train_factory):
    # Get training args class depending on the chosen model pipeline

    train_args = train_factory.create_argument_class(args.model_pipeline,
                            args.act_fun,
                            args.batch_size, 
                            args.filters,
                            args.detach_fusion_layers,
                            args.iterations,
                            args.loss,
                            args.norm_method,
                            args.optimizer,
                            args.pool_method,
                            args.pre_train,
                            args.retrain,
                            args.split,
                            args.crop_size,
                            args.mode,
                            args.pre_processing_pipeline,
                            args.softmax_layer,
                            args.classification_loss)

    # Training parameters used for all the iterations/crop options given.                         
    log.info(train_args)
    return train_args


def get_model_config(log, train_args, num_gpus):
    # Get training settings
    
    # Parsing the model configurations - get CNN (double encoder U-Net). WARNING: Double 'decoder', not encoder.
    model_config = {'architecture': train_args.get_arch_args(),
                    'batch_size': train_args.batch_size,
                    'batch_size_auto': 2,
                    'label_type': "distance", # NOTE: Fixed param.
                    'loss': train_args.loss,
                    'classification_loss': train_args.classification_loss,
                    'num_gpus': num_gpus,
                    'optimizer': train_args.optimizer
                    }

    log.info(f"Model configuration {model_config}")
    return model_config


def set_up_training_loops(log, args, path_data, trainset_name, path_models, model_config, net, num_gpus, device):
    # Loop to iterate over the different trained/re-trained architectures.

    for idx, crop_size in enumerate(args.crop_size): # Cicle over multiple 'crop_size' if provided
        model_name = '{}_{}_{}_{}_{}_{}'.format(trainset_name, args.mode, args.split, crop_size, args.pre_processing_pipeline, args.arch)
        log.info(f"--- The '{idx + 1}' model used is {model_name} ---")

        # Train multiple models - It helps to distinguish between model apparently equal (but with internal architecture different)
        for i in range(args.iterations): 
        
            run_name = unique_path(path_models, model_name + '_{:02d}.pth').stem
            log.debug(f"-- Run name: {run_name} - Iteration: {i} --")
            # Update the current model configurations - there will be other updates but the core parameters will remain the same (e.g. architecture)
            model_config['run_name'] = run_name

            if args.retrain: # TODO: To change/optimize this sub-condition. 
                old_model = Path(__file__).parent / args.retrain
                if get_file(old_model.parent / "{}.json".format(old_model.stem))['architecture'][-1] != model_config['architecture'][-1]:
                    raise Exception('Architecture of model to retrain does not match.')
                # Get weights of trained model to retrain
                print("Load models of {}".format(old_model.stem))
                net = get_weights(net=net, weights=str('{}.pth'.format(old_model)), num_gpus=num_gpus, device=device)
                model_config['retrain_model'] = old_model.stem

            # Pre-training of the Encoder in autoencoder style
            model_config['pre_trained'] = False
            if args.pre_train:

                if args.mode != 'GT' or len(args.cell_type) > 1:
                    raise Exception('Pre-training only for GTs and for single cell type!')

                # Get CNN (U-Net without skip connections)
                net_auto = create_model_architecture(log, args.pre_train, model_config, device, num_gpus)

                # Load training and validation set
                data_transforms_auto = augmentors(label_type='auto', min_value=0, max_value=65535)
                datasets = AutoEncoderDataset(data_dir=path_data / args.cell_type[0],
                                          train_dir=path_data / "{}_{}_{}_{}".format(trainset_name, args.mode, args.split, crop_size),
                                          transform=data_transforms_auto)

                # Train model (pre-training configurations)
                train_auto(net=net_auto, dataset=datasets, configs=model_config, device=device,  path_models=path_models)

                # Load best weights and load best weights into encoder before the fine-tuning.
                net_auto = get_weights(net=net_auto, weights=str(path_models / '{}.pth'.format(run_name)), num_gpus=num_gpus, device=device)
                pretrained_dict, net_dict = net_auto.state_dict(), net.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}  # 1. filter unnecessary keys
                net_dict.update(pretrained_dict)  # 2. overwrite entries
                net.load_state_dict(net_dict)  # 3. load the new state dict
                model_config['pre_trained'] = True
                del net_auto
            
            # Load training and validation set - this is the train (or fine-tuning after the pre-training phase).
            data_transforms = augmentors(label_type = model_config['label_type'], min_value=0, max_value=65535) # NOTE: min_value and max_value fixed params.
            model_config['data_transforms'] = str(data_transforms)
            dataset_name = "{}_{}_{}_{}".format(trainset_name, args.mode, args.split, crop_size)
            log.debug(f".. Reading dataset: {dataset_name} ..")
            
            # In the original script it was implemented the 'all' dataset plus ST option.
            datasets = {x: CellSegDataset(root_dir=path_data / dataset_name, mode=x, transform=data_transforms[x])
                        for x in ['train', 'val']}

            # Train loop with the chosen architecture.
            best_loss = train(log=log, net=net, datasets=datasets, config=model_config, device=device, path_models=path_models)

             # Fine-tune with cosine annealing for Ranger models - taken the current best model just trained.
            if model_config['optimizer'] == 'ranger':
                # NOTE: No-update on training configurations.
                net = create_model_architecture(log, args.pre_train, model_config, device, num_gpus)
                # Get best weights as starting point
                net = get_weights(net=net, weights=str(path_models / '{}.pth'.format(run_name)), num_gpus=num_gpus, device=device)
                # Train further
                _ = train(net=net, datasets=datasets, config=model_config, device=device, path_models=path_models, best_loss=best_loss)

            # Write information to json-file - consider to pass even the train_args.
            write_train_info(configs=model_config, path=path_models)
    return None


def set_up_training():
    """ Main function to set up paths, datasets and training pipelines
    """
    args, log = set_up_args_and_folder()

    log.info(f"Args: {args}") # Print overall args
    device, num_gpus = set_device() # Set device: cpu or single-gpu usage
    log.info(f"System detected {device} device and {num_gpus} GPUs available.")
    log.info(f">>>   Training: pre-processing {args.pre_processing_pipeline} model {args.model_pipeline}   <<<")

    # Load paths
    path_data = Path(args.train_images_path)
    path_models = Path(args.models_folder) # Train all models found here.
    
    # TODO: Make modular - for now just take the first dataset available indicated by the parameter.
    args.dataset = args.dataset[0] # TODO: To fix the temporary 'work around'.
    cell_type = Path(args.dataset)
    
    # Double check if both the training data folder and the specific dataset exist
    if check_path(log, path_data) and check_path(log, join(path_data, cell_type)):
        trainset_name = args.dataset # 'args.dataset' used as cell type
    set_up_training_set(log, args, path_data, cell_type)

    # If it is desired to just create the training set
    if not args.train_loop:
        log.info(f">>> Creation of the trainining dataset scripts ended correctly <<<")
        return None # Exit the script

    # Instantiate the training factory class
    factory = train_factory()

    # Parse the training arguments and settings for the specific model pipeline
    train_args = get_training_args_class(log, args, factory)
    model_config = get_model_config(log, train_args, num_gpus)
    net = create_model_architecture(log, model_config, device, num_gpus, train_args.pre_train)

    # Set up and execute the actual trainig
    set_up_training_loops(log, train_args, path_data, trainset_name, path_models, model_config, net, num_gpus, device)
    log.info(">>> Training script ended correctly <<<")
    return


if __name__ == "__main__":
    set_up_training()