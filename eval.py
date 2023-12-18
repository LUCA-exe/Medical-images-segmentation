"""eval.py

This is the evaluation main function that call inference and the metrics computation.
"""
import os
from os.path import join, exists
from collections import defaultdict
from utils import create_logging,  set_device, EvalArgs
from parser import get_parser, get_processed_args
from inference.inference import inference_2d_ctc
from net_utils.metrics import count_det_errors, ctc_metrics


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
    log.info(f">>>   Evaluation: model {args.model_pipeline} post-processing {args.post_processing_pipeline} metrics {args.eval_metric} <<<")

    # Load paths
    path_data = join(args.train_images_path, args.dataset)
    path_models = args.models_folder # Eval all models found here.
    path_best_models = args.save_model # Save best model files/metrics here
    path_ctc_metric = args.evaluation_software

    if not exists(path_data):
        log.info(f"Warning: the '{path_data}' provided is not existent! Interrupting the program...")
        raise ValueError("The '{path_data}' provided is not existent")

    if args.eval_metric == 'software': # Check wich set of metrics you want to use
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
    
    # NOTE: For now it is implemented evaluation for one dataset
    if args.model_pipeline == 'kit-ge': # Call inference from the KIT-GE-(2) model's method
        kit_ge_inference_loop(log, models, path_models, train_sets, path_data, device, args.scale, args)
    
    else: # Call other inference loop ..
        raise NotImplementedError(f"Other inference options not implemented yet ..")

    # this current params loop is specific for kit-ge pipeline..
    '''for model in models: # Loop over all the models found
        model_path = os.path.join(path_models, model) # Create model path
        for th_seed in args.th_seed:# Go through thresholds
            for th_cell in args.th_cell:
                for train_set in train_sets:
                        log.info(f'>>> Evaluate {model} on {path_data}_{train_set}: th_seed: {th_seed}, th_cell: {th_cell}')

                # Set up current results folder for the segmentation maps
                path_seg_results = os.path.join(path_data, f"{train_set}_RES_{model.split('.')[0]}_{th_seed}_{th_cell}")
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

                if args.model_pipeline == 'kit-ge': # Call inference from the KIT-GE-(2) model's method
                    # Inference on the chosen train set
                    inference_2d_ctc(log=log, model=model_path,
                                data_path=os.path.join(path_data, train_set),
                                result_path=path_seg_results,
                                device=device,
                                batchsize=args.batch_size,
                                args=eval_args,
                                num_gpus=1, # Warning: Fixed for now at 1.
                                model_pipeline=args.model_pipeline,
                                post_processing_pipeline=args.post_processing_pipeline) 
                else:
                    raise NotImplementedError(f"Other inference options not implemented yet ..")

                # TODO: Gather metrics and evaluate dict for gathering metrics results'''
    
    log.info(">>> Evaluation script ended correctly")


# TODO: Implementing kit-ge inference loop.
def kit_ge_inference_loop(log, models, path_models, train_sets, path_data, device, scale_factor, args):

    # TODO: Get dict to store results - implementing .. .. ..
    results = defaultdict(dict)

    # NOTE: For now it is implemented evaluation for one dataset - this current params loop is specific for kit-ge pipeline..
    for model in models: # Loop over all the models found
        model_path = os.path.join(path_models, model) # Create model path
        for th_seed in args.th_seed:# Go through thresholds
            for th_cell in args.th_cell:
                for train_set in train_sets:

                    log.info(f'> Evaluate {model} on {path_data}_{train_set}: th_seed: {th_seed}, th_cell: {th_cell}')

                    # Set up current results folder for the segmentation maps
                    path_seg_results = os.path.join(path_data, f"{train_set}_RES_{model.split('.')[0]}_{th_seed}_{th_cell}")
                    log.info(f"The result of the current evaluation will be saved in '{path_seg_results}'")
                    os.makedirs(path_seg_results, exist_ok=True)

                    # Get post-processing settings
                    eval_args = EvalArgs(args.post_processing_pipeline,
                                            th_cell=float(th_cell), 
                                            th_seed=float(th_seed),
                                            apply_clahe=args.apply_clahe,
                                            scale=scale_factor,
                                            cell_type=args.cell_type,
                                            save_raw_pred=args.save_raw_pred,
                                            artifact_correction=args.artifact_correction,
                                            apply_merging=args.apply_merging)
                    
                    # Debug specific args for the current run.
                    log.debug(eval_args)

                    # Inference on the chosen train set
                    args_used = inference_2d_ctc(log=log, model=model_path,
                                data_path=os.path.join(path_data, train_set),
                                result_path=path_seg_results,
                                device=device,
                                batchsize=args.batch_size,
                                args=eval_args,
                                num_gpus=1, # Warning: Fixed for now at 1.
                                model_pipeline=args.model_pipeline,
                                post_processing_pipeline=args.post_processing_pipeline) 

                    if args.eval_metric == 'software':
                        seg_measure, det_measure = ctc_metrics(path_data=path_data,
                                                                path_results=path_seg_results,
                                                                path_software=args.evaluation_software,
                                                                subset=train_set,
                                                                mode=args.mode)
                    else:
                        raise NotImplementedError(f"Other metrics not implemented yet ..")
            
                    # Save parameters of the model and post_processing
                    results[f"model.split('.')[0]"] = {'model':args_used, 
                                                    'post_processing':eval_args,
                                                    f'train_set':train_set,
                                                    'results':None} # save metrics
    
    return None


if __name__ == "__main__":
    main()