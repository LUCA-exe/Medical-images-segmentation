""" parser.py:

- Configuration of the parameters.
"""

from argparse import ArgumentParser

def get_parser():
    """ Standard function to load the parameters

        Returns:
            Object: parser
    """
    parser = ArgumentParser(description="Medical images segmentation")

    # Path args
    parser.add_argument("--train_images_path",
                            default="training_data/",
                            type=str,
                            help="Path to the train images dataset")

    parser.add_argument("--test_images_path",
                            default="test_data/",
                            type=str,
                            help="Path to the test images dataset")

    parser.add_argument("--dataset", # Default parameter is my traind dataset
                            default="Fluo-E2DV-train",
                            type=str,
                            help="Which folder to access")

    parser.add_argument("--models_folder",
                            default="./models/all/",
                            type=str,
                            help="Which folder to access for train/test models")

    parser.add_argument("--save_model",
                            default="./models/best/",
                            type=str,
                            help="Which folder to access for saving best model files/metrics")

    parser.add_argument("--eval_metric",
                            default="software", # TODO: implement more options.
                            type=str,
                            help="Str used to decide which metrics use for evaluation")

    parser.add_argument("--evaluation_software",
                            default="./net_utils/evaluation_software/", # Specific folder to save the evaluation software 
                            type=str,
                            help="Path to access for loading the evaluation software of Cell Tracking Challenge")

    parser.add_argument("--download",
                            default=False,
                            type=bool,
                            help="Boolean value to check for download online single or multiple datasets")

    # What dataset to download (from CTC website)
    parser.add_argument("--download_dataset",
                            default = "all", # TODO: Expand the dataset options from other challenges/websites
                            type=str,
                            help="Name of the dataset to download (else 'all')")
    
    # Starting args used for the image characteristics gathering

    # TODO: Kept fixed as arguments.. maybe should be extracted from a '*.json' file with a finetuned dimension for every seen dataset..
    parser.add_argument("--cell_dim",
                            default = 7000, # My dataset has an avg of 4000 EVs dim in pixels (It depends on the resolution).
                            type=int,
                            help="Min. dimension (in pixels) to consider for gathering cells stats/signals")

    parser.add_argument("--max_images", 
                            default = 15,
                            type=int,
                            help="Max. number of images to take for the gathering of signals (for every folder separately)")
     
    parser.add_argument("--compute_signals", # Gather signals for a dataset 
                            default = False,
                            type = bool,
                            help="Compute signals for the chosen dataset")

    parser.add_argument("--compare_signals", # Compare signals for all datasets (both box-plots and line-plots) 
                            default = False,
                            type = bool,
                            help="Compare signals for all the dataset with computed signals.")

    # Inference/Evaluation args
    parser.add_argument("--models_split", default="models/trained", type=str, help="Path to fetch the chosen model")
    parser.add_argument("--models_name", default="none", type=str, help="model's name to fetch")
    parser.add_argument('--apply_merging', '-am', default=False, action='store_true', help='Merging post-processing')
    parser.add_argument('--artifact_correction', '-ac', default=False, action='store_true', help='Artifact correction')
    parser.add_argument('--batch_size', '-bs', default=1, type=int, help='Batch size')
    #parser.add_argument('--cell_type', '-ct', nargs='+', required=True, help='Cell type(s)')
    parser.add_argument('--mode', '-m', default='GT', type=str, help='Ground truth type / evaluation mode')
    #parser.add_argument('--models', required=True, type=str, help='Models to evaluate (prefix)')
    parser.add_argument('--multi_gpu', '-mgpu', default=False, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--apply_clahe', '-acl', default=False, action='store_true', help='CLAHE pre-processing')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save some raw predictions')
    parser.add_argument('--scale', '-sc', default=1, type=float, help='Scale factor (0: get from trainset info.json') # json file with per-dataset parameters
    parser.add_argument('--subset', '-s', default='01', type=str, help='Subset to evaluate on') # Possible options: [01, 02,01+02]
    parser.add_argument('--th_cell', '-tc', default=0.07, nargs='+', help='Threshold for adjusting cell size')
    parser.add_argument('--th_seed', '-ts', default=0.45, nargs='+', help='Threshold for seeds')

    args = parser.parse_args()
    return args


def get_processed_args(args):
    """ Process the args from config file and console to usable format

        Args:
            output (Tensor): predicted probabilities, higher = more confidence

        Returns:
            args (dict):
    """
    # Process args before train/eval
    if args.subset == '01+02':
        args.subset = ['01','02']
    else:
        args.subset = [args.subset]

    # Providing a gridsearch for finetunable parameters - kept from the original repository.
    if not isinstance(args.th_seed, list):
        args.th_seed = [args.th_seed]
    if not isinstance(args.th_cell, list):
        args.th_cell = [args.th_cell]

    # TODO: Check if this method is the best
    args.cell_type = args.dataset

    return args











