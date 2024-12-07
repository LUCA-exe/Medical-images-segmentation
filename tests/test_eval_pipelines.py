"""This module test the main methods and pipelines for the evaluation of NNs.

Specifically, This testing module provide the following main test features:
- Model evaluation loops.
- Post-processing methods.

To test just the sub-functions:
...> python -m pytest -v --run-sub tests/test_eval_pipelines.py

To test just the entire pipeline:
...> python -m pytest -v --run-pipeline tests/test_eval_pipelines.py

To visualize the 'stdout' (console prints) add -s as parameter when launching the test.
In some tests it will uses the *.npy files listed in the ./tests/README.txt file.
"""
import pytest
import subprocess
from os.path import join
import os 

from test_train_pipelines import update_default_args
from utils import load_environment_variables, set_current_run_folders, \
    create_logging, set_device, check_path, read_json_file, check_and_download_evaluation_software
from parser import get_parser, get_processed_args
from eval import du_inference_loop, tu_inference_loop

def mock_evaluation_set_up(args) -> None:
    """The called functions will to set up paths, load model and evaluate the inferred images.

    In this evaluations (mock) set-up is not needed the parser methods for the args.
    """
    load_environment_variables()
    set_current_run_folders()
    log = create_logging() 
    log.info(f"**** THIS LOG FILE IS CREATED USING the mock_valuation_set_up(...) method in the ./tests/test_eval_pipelines.py module ****")

    env = {} # TODO: Load this from a '.json' file environment parameters
    env['logger'] = log # Move the object through 'env' dict
    log.info(f"Args: {args}")
    log.debug(f"Env varibles: {env}")
    device, num_gpus = set_device()
    log.info(f">>>   Evaluation: model {args['model_pipeline']} post-processing {args['post_processing_pipeline']} metrics {args['eval_metric']} <<<")

    # Load paths
    args["dataset"] = args["dataset"][0] # TODO: To fix!!!
    path_data = join(args["train_images_path"], args["dataset"])
    path_models = args["models_folder"]  # Eval all models found here.

    # NOTE: Unusued for now.
    path_best_models = args["save_model"]
    path_ctc_metric = args["evaluation_software_path"]
    check_path(log, path_data)
    if args["eval_metric"] == 'software': # Check wich set of metrics you want to use.
        if not os.path.isdir(path_ctc_metric) or len(os.listdir(path_ctc_metric)) < 2: # Check if evaluation software is available
            check_and_download_evaluation_software(log, path_ctc_metric)
            
            # NOTE: Temporary solution.
            software_path = os.path.join('net_utils', 'evaluation_software')
            captured_stdout = subprocess.run(["chmod", "-R", "755", software_path], capture_output=True)
            if captured_stdout.stdout != "b''":
                raise ValueError(f"The sub-process used to change permission of the evaluation sofware has failed!")
    models = [model for model in os.listdir(args["models_folder"]) if model.endswith('.pth')]
    if len(models) < 1:
        raise ValueError(f"The are no *.pth files inside the folder {args['models_folder']}!")

    # Load the paths in the log files
    log.info(f"Dataset folder to evaluate: {path_data}")
    log.info(f"Folder used to fetch models: {path_models}")
    log.info(f"Folder used to save models performances/files: {path_data}")
    if path_ctc_metric != 'none': # In case of 'none' (str) use custom metrics on the script.
        log.info(f"Evaluation software folder: {path_ctc_metric}")

    scores = [] # Temporary list to keep the evaluation results
    train_sets = [args["split"]]  # List of subfolder to eval.

    # NOTE: For now it is implemented evaluation for one dataset
    if args["post_processing_pipeline"] == 'dual-unet': # Call inference from the KIT-GE-(2) model's method
        du_inference_loop(log, models, path_models, train_sets, path_data, device, num_gpus, args)

    # TODO: Finish to test
    elif args["post_processing_pipeline"] == 'original-dual-unet':
        tu_inference_loop(log, models, path_models, train_sets, path_data, device, num_gpus, args)
    
    else: # Call other inference loop ..
        raise NotImplementedError(f"Other inference options for testing are not implemented yet ..")
    log.info(">>> Evaluation script ended correctly <<<")


class TestMockTrainPipelines:
    """This class contains functions to simulate and test the evaluation pipeline.
    """

    @pytest.mark.pipeline
    def test_eval_pipeline(self):
        """This function test all the methods from the loading of the model
        to the final post-processing implementation.
        """
        default_args = read_json_file("./tests/mock_eval_args.json")
        test_arguments = [
            {"models_folder": "./tests/model_configs/", "model_pipeline": "dual-unet", "post_processing_pipeline": "dual-unet", "dataset": ["Mock-E2DV-train"], "is_unit_test": True},
            #{"model_pipeline": "original-dual-unet", "dataset": "Mock-E2DV-train", "crop_size": 640, "filters": [64, 128]}
        ]

        for test_args in test_arguments:
            run_parameters = update_default_args(default_args, test_args)
            mock_evaluation_set_up(run_parameters)

            