"""eval.py

This is the evaluation main function that call inference and the metrics computation.
"""
import os
from os.path import join, exists
from collections import defaultdict
from utils import create_logging, set_device, load_environment_variables, set_current_run_folders, check_path, eval_factory
from parser import get_parser, get_processed_args
from inference.inference import inference_2d # Main inference loop
from net_utils.metrics import count_det_errors, ctc_metrics
from net_utils.utils import save_metrics

SOFTWARE_DET_FILE = "DET_log.txt"

def main():
    """ Main function to set up paths, load model and evaluate the inferred images.
    """
    load_environment_variables()
    set_current_run_folders()
    log = create_logging() # Set up 'logger' object 

    args = get_parser() # Set up dict arguments
    args = get_processed_args(args)

    env = {} # TODO: Load this from a '.json' file environment parameters
    env['logger'] = log # Move the object through 'env' dict

    log.info(f"Args: {args}") # Print overall args 
    log.debug(f"Env varibles: {env}")
    device, num_gpus = set_device() # Set device: cpu or single-gpu usage.
    
    log.info(f">>>   Evaluation: model {args.model_pipeline} post-processing {args.post_processing_pipeline} metrics {args.eval_metric} <<<")

    # Load paths
    args.dataset = args.dataset[0] # TODO: To fix!!!
    path_data = join(args.train_images_path, args.dataset)
    path_models = args.models_folder # Eval all models found here.
    path_best_models = args.save_model # Save best model files/metrics here.
    path_ctc_metric = args.evaluation_software_path

    check_path(log, path_data)

    if args.eval_metric == 'software': # Check wich set of metrics you want to use.
        if not os.path.isdir(path_ctc_metric): # Check if evaluation software is available
            raise Exception('No evaluation software found. Run the script download_data.py')

    # Load all models in the chosen folder
    models = [model for model in os.listdir(args.models_folder) if model.endswith('.pth')]

    # Load the paths in the log files
    log.debug(f"Dataset folder to evaluate: {path_data}")
    log.debug(f"Folder used to fetch models: {path_models}")
    log.debug(f"Folder used to save models performances/files: {path_data}") # Saving the results on the folder of the evaluated dataset
    if path_ctc_metric != 'none': # In case of 'none' (str) use custom metrics on the script.
        log.debug(f"Evaluation software folder: {path_ctc_metric}")

    scores = [] # Temporary list to keep the evaluation results
    train_sets = args.subset # List of subfolder to eval: already parsed from args

    # NOTE: For now it is implemented evaluation for one dataset
    if args.post_processing_pipeline == 'dual-unet': # Call inference from the KIT-GE-(2) model's method
        du_inference_loop(log, models, path_models, train_sets, path_data, device, num_gpus, args)

    if args.post_processing_pipeline == 'fusion-dual-unet':
        fdu_inference_loop(log, models, path_models, train_sets, path_data, device, num_gpus, args)

    elif args.post_processing_pipeline == 'triple-unet':
        tu_inference_loop(log, models, path_models, train_sets, path_data, device, num_gpus, args)

    # TODO: Finish to test
    elif args.post_processing_pipeline == 'original-dual-unet':
        tu_inference_loop(log, models, path_models, train_sets, path_data, device, num_gpus, args)
    
    else: # Call other inference loop ..
        raise NotImplementedError(f"Other inference options not implemented yet ..")
        
    log.info(">>> Evaluation script ended correctly <<<")
    return None


def du_inference_loop(log, models, path_models, train_sets, path_data, device, num_gpus, args):
    # Implementing kit-ge inference loop if 'post-processing' selected is theirs.

    result_dict = {}
    curr_experiment = 0 # Simple counter of the args. combination
    eval_f = eval_factory()

    # NOTE: For now it is implemented evaluation for one dataset - this current params loop is specific for kit-ge pipeline..
    for model in models: # Loop over all the models found
        model_path = os.path.join(path_models, model) # Create model path
        for th_seed in args["th_seed"]:# Go through thresholds
            for th_cell in args["th_cell"]:
                for train_set in train_sets:

                    log.info(f'> Evaluate {model} on {path_data}_{train_set}: th_seed: {th_seed}, th_cell: {th_cell}')

                    # Set up current results folder for the segmentation maps
                    path_seg_results = os.path.join(path_data, f"{train_set}_RES_{model.split('.')[0]}_{th_seed}_{th_cell}")
                    log.info(f"The result of the current evaluation will be saved in '{path_seg_results}'")
                    os.makedirs(path_seg_results, exist_ok=True)

                    eval_class_args = eval_f.create_argument_class(args["post_processing_pipeline"],
                                                    float(th_cell), 
                                                    float(th_seed),
                                                    args["apply_clahe"],
                                                    args["scale"],
                                                    args["dataset"],
                                                    args["save_raw_pred"],
                                                    args["artifact_correction"],
                                                    args["apply_merging"])
                    
                    # Debug specific args for the current run.
                    log.debug(eval_class_args)

                    # Inference on the chosen train set
                    args_used = inference_2d(log=log, model_path=model_path,
                                data_path=os.path.join(path_data, train_set),
                                result_path=path_seg_results,
                                device=device,
                                num_gpus = num_gpus,
                                batchsize=args.batch_size,
                                args=eval_class_args) 

                    if args.eval_metric == 'software':
                        seg_measure, det_measure = ctc_metrics(path_data=path_data,
                                                                path_results=path_seg_results,
                                                                path_software=args.evaluation_software_path,
                                                                subset=train_set,
                                                                mode=args.mode)
                        
                        _, so, fnv, fpv = count_det_errors(os.path.join(path_seg_results, SOFTWARE_DET_FILE))

                    else:
                        raise NotImplementedError(f"Other metrics not implemented yet ..")

                    # NOTE: Every post-processing pipeline have internal specific args - custom aggregation later (both for visualization/tranform in '*.csv' file).
                    result_dict[curr_experiment] = {'model':model, 'th_cell': str(th_cell), 'th_seed': str(th_seed), 'train_set': str(train_set), 'SEG':seg_measure, 'DET': det_measure, 'SO':so, 'FNV':fnv, 'FPV': fpv}
                    curr_experiment += 1
                   
    # Save the metrics - It will update the file if there is already an "*.json" with the same name.
    result_file_name = f"{args['model_pipeline']}_{args['post_processing_pipeline']}_{path_data.split('/')[-1]}_eval_results"
    save_metrics(log, result_dict, path_data, name = result_file_name)
    return None


def fdu_inference_loop(log, models, path_models, train_sets, path_data, device, num_gpus, args):
    # Implementing fusion kit-ge inference loop

    result_dict = {}
    curr_experiment = 0 # Simple counter of the args. combination
    eval_f = eval_factory()

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

                    eval_class_args = eval_f.create_argument_class(args.post_processing_pipeline,
                                                    float(th_cell), 
                                                    float(th_seed),
                                                    args.apply_clahe,
                                                    args.scale,
                                                    args.dataset,
                                                    args.save_raw_pred,
                                                    args.artifact_correction,
                                                    args.apply_merging,
                                                    args.fusion_overlap)
                    
                    # Debug specific args for the current run.
                    log.debug(eval_class_args)

                    # Inference on the chosen train set
                    args_used = inference_2d(log=log, model_path=model_path,
                                data_path=os.path.join(path_data, train_set),
                                result_path=path_seg_results,
                                device=device,
                                num_gpus = num_gpus,
                                batchsize=args.batch_size,
                                args=eval_class_args) 

                    if args.eval_metric == 'software':
                        seg_measure, det_measure = ctc_metrics(path_data=path_data,
                                                                path_results=path_seg_results,
                                                                path_software=args.evaluation_software_path,
                                                                subset=train_set,
                                                                mode=args.mode)
                        
                        _, so, fnv, fpv = count_det_errors(os.path.join(path_seg_results, SOFTWARE_DET_FILE))

                    else:
                        raise NotImplementedError(f"Other metrics not implemented yet ..")

                    # NOTE: Every post-processing pipeline have internal specific args - custom aggregation later (both for visualization/tranform in '*.csv' file).
                    result_dict[curr_experiment] = {'model':model, 'th_cell': str(th_cell), 'th_seed': str(th_seed), 'train_set': str(train_set), 'SEG':seg_measure, 'DET': det_measure, 'SO':so, 'FNV':fnv, 'FPV': fpv}
                    curr_experiment += 1
                   
    # Save the metrics - It will update the file if there is already an "*.json" with the same name.
    result_file_name = f"{args.model_pipeline}_{args.post_processing_pipeline}_{path_data.split('/')[-1]}_eval_results"
    save_metrics(log, result_dict, path_data, name = result_file_name)
    return None


# TODO: Work in progress - to test
def tu_inference_loop(log, models, path_models, train_sets, path_data, device, num_gpus, args):

    result_dict = {}
    curr_experiment = 0 # Simple counter of the args. combination
    eval_f = eval_factory()

    # NOTE: For now it is implemented evaluation for one dataset - this current params loop is specific for kit-ge pipeline..
    for model in models: # Loop over all the models found
        model_path = os.path.join(path_models, model) # Create model path
        for train_set in train_sets:

            log.info(f'> Evaluate {model} on {path_data}_{train_set}')

            # Set up current results folder for the segmentation maps
            path_seg_results = os.path.join(path_data, f"{train_set}_RES_{model.split('.')[0]}")
            log.info(f"The result of the current evaluation will be saved in '{path_seg_results}'")
            os.makedirs(path_seg_results, exist_ok=True)

            # Get post-processing settings
            eval_class_args = eval_f.create_argument_class(args.post_processing_pipeline,
                                                    args.apply_clahe,
                                                    args.scale,
                                                    args.dataset,
                                                    args.save_raw_pred,
                                                    args.artifact_correction,
                                                    args.apply_merging)
            
            # Debug specific args for the current run.
            log.debug(eval_class_args)

            # Inference on the chosen train set
            args_used = inference_2d(log=log, 
                        model_path=model_path,
                        data_path=os.path.join(path_data, train_set),
                        result_path=path_seg_results,
                        device=device,
                        num_gpus = num_gpus,
                        batchsize=args.batch_size,
                        args=eval_class_args) 

            if args.eval_metric == 'software':
                seg_measure, det_measure = ctc_metrics(path_data=path_data,
                                                        path_results=path_seg_results,
                                                        path_software=args.evaluation_software_path,
                                                        subset=train_set,
                                                        mode=args.mode)
                
                _, so, fnv, fpv = count_det_errors(os.path.join(path_seg_results, SOFTWARE_DET_FILE))

            else:
                raise NotImplementedError(f"Other metrics not implemented yet ..")

            # NOTE: Every post-processing pipeline have internal specific args - custom aggregation later (both for visualization/tranform in '*.csv' file).
            result_dict[curr_experiment] = {'model':model, 'train_set': str(train_set), 'SEG':seg_measure, 'DET': det_measure, 'SO':so, 'FNV':fnv, 'FPV': fpv}
            curr_experiment += 1
                   
    # Save the metrics - It will update the file if there is already an "*.json" with the same name.
    result_file_name = f"{args.model_pipeline}_{args.post_processing_pipeline}_{path_data.split('/')[-1]}_eval_results"
    save_metrics(log, result_dict, path_data, name = result_file_name)
    return None


if __name__ == "__main__":
    main()