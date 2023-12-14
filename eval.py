"""eval.py

This is the evaluation main function that call inference and the metrics computation.
"""
import os
from os.path import join, exists
from utils import create_logging,  set_device, EvalArgs
from parser import get_parser, get_processed_args
from inference.inference import inference_2d_ctc


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
    device = set_device() # Set device: cpu or single-gpu usage
    log.debug(f">>>   Evaluation   <<<")

    # Load paths
    path_data = join(args.train_images_path, args.dataset)
    path_models = args.models_folder # Eval all models found here.
    path_best_models = args.save_model # Save best model files/metrics here
    path_ctc_metric = args.evaluation_software

    if not exists(path_data):
        log.debug(f"Warning: the '{path_data}' provided is not existent! Interrupting the program...")
        raise ValueError("The '{path_data}' provided is not existent")

    # Check wich set of metrics you want to use
    if args.eval_metric == 'software':
        if not os.path.isdir(path_ctc_metric): # Check if evaluation software is available
            raise Exception('No evaluation software found. Run the script download_data.py')

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
    scale_factor = args.scale # TODO: Get scale from training dataset info if not stated otherwise
        
    # NOTE: For now it is implemented evaluation for one dataset
    for model in models: # Loop over all the models found
        model_path = os.path.join(path_models, model) # Create model path
        for th_seed in args.th_seed:# Go through thresholds
            for th_cell in args.th_cell:
                for train_set in train_sets:
                        log.info(f'---- ---- ----\nEvaluate {model} on {path_data}_{train_set}: th_seed: {th_seed}, th_cell: {th_cell}')

                # Set up current results folder
                path_seg_results = os.path.join(path_data, f"{train_set}_RES_{model}_{th_seed}_{th_cell}")
                log.info(f"The result of the current evaluation will be saved in '{path_seg_results}'")
                os.makedirs(path_seg_results, exist_ok=True)

                # Get post-processing settings
                eval_args = EvalArgs(th_cell=float(th_cell), th_seed=float(th_seed),
                                        apply_clahe=args.apply_clahe,
                                        scale=scale_factor,
                                        cell_type=args.cell_type,
                                        save_raw_pred=args.save_raw_pred,
                                        artifact_correction=args.artifact_correction,
                                        apply_merging=args.apply_merging)

                log.debug(eval_args)
                
                # Inference on the chosen train set
                inference_2d_ctc(log=log, model=model_path,
                             data_path=os.path.join(path_data, train_set),
                             result_path=path_seg_results,
                             device=device,
                             batchsize=args.batch_size,
                             args=eval_args,
                             num_gpus=1) # Warning: Fixed for now at 1.

                # TODO: Gather metrics and evaluate dict for gathering metrics results


if __name__ == "__main__":
    main()