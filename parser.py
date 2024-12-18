""" parser.py:

- Configuration of the parameters.
"""

from argparse import ArgumentParser

def get_parser():
    """ Standard function to load the parameters

        Returns:
            Object: parser
    """
    parser = ArgumentParser(description="Medical images segmentation arguments")

    # Path arguments use trhoughout the repository

    parser.add_argument("--train_images_path",default="training_data/", type=str, help="Path to the train images dataset")
    parser.add_argument("--test_images_path", default="test_data/", type=str, help="Path to the test images dataset")
    # Default parameter is my traind dataset (Experiment 01 annotation 02)
    parser.add_argument("--dataset", default="Fluo-E2DV-train", nargs='+', type=str, help="Which dataset folder to access (ca be multiple)")
    parser.add_argument("--models_folder", default="./models/all/", type=str, help="Which folder to access for train/val/test models")
    # TODO: Decided how to use this folder - require manual supervision to analyze its results
    parser.add_argument("--save_model", default="./models/best/", type=str, help="Which folder to access for saving best model files/metrics")
    # For now kept as default the CTC evaluation DET and SEG scores -  can be implemented more options
    parser.add_argument("--eval_metric", default="software", type=str, help="Str used to decide which metrics use for evaluation")
    # Specific folder to save the evaluation software - should be treated as const
    parser.add_argument("--evaluation_software_path", default="./net_utils/evaluation_software/", type=str, help="Path to access for loading the evaluation software of Cell Tracking Challenge")
    # TODO: Expand the dataset options from other challenges/websites - what dataset to download (from CTC website)
    parser.add_argument("--dataset_to_download", default = "all", type=str, help="Name of the dataset to download (else 'all')")
    
    # Parameters used for the image characteristics gathering (analyze image properties)

    parser.add_argument("--split_signals", default=False, action="store_true", help='Just for my Datasets, split the charateristics between EVs and Cells')
    # TODO: Kept fixed as arguments.. maybe should be extracted from a '*.json' file with a finetuned dimension for every seen dataset..
    parser.add_argument("--cell_dim", type=int, default=6000, help='Value used as a threshold between the two type of cells')
    # NOTE: My dataset has an avg of 4000 EVs dim in pixels - It depends on the resolution.
    parser.add_argument("--max_images", default = 15, type=int, help="Max. number of images to take for the gathering of signals (for every folder ('01', '02') separately)")
    parser.add_argument('--compare_signals', default=False, action="store_true", help='Aggregate and compare signals for all the dataset with computed signals.')
    parser.add_argument('--compute_data_properties', default=False, action="store_true", help='Compute the signals of the indicated dataset or datasets')
    
   
    # Mixed pipelines parameters (pre-processing for training/models/post processing methods for evaluation)

    parser.add_argument("--pre_processing_pipeline", default="kit-ge", type=str, help="Chosing what pre-processing operations/pipeline to do")
    parser.add_argument('--train_loop', '-tl', default=False, action="store_true", help='If True execute the training loop after the dataset creation')
    parser.add_argument("--model_pipeline", default="dual-unet", type=str, help="String to chose what models to build")
    parser.add_argument("--post_processing_pipeline", default="dual-unet", type=str, help="Chosing what post-processing operations/pipeline to do")

    # Dataset pre-processing args - for now just KIT-GE post-processing pipeline it is implemented

    parser.add_argument('--min_a_images', '-mai', default=30, type=int, help="Minimum number of 'A' annotated patches, if less take even the 'B' quality patches")
    parser.add_argument('--crop_size', '-cs', default=320, type=int, nargs='+', help="Crop size for creating the dataset")
    
    # Training args - for now just KIT-GE post-processing pipeline args it is implemented

    parser.add_argument('--act_fun', '-af', default='relu', type=str, help='Activation function')
    parser.add_argument('--filters', '-f', nargs=2, type=int, default=[64, 1024], help='Filters for U-net')
    # NOTE: This arg. is parsed just if the model pipeline is the "dual-unet".
    parser.add_argument('--detach_fusion_layers', '-dfl', default=False, action="store_true", help='If it is allowed the back-propagation from the fusion layers')
    parser.add_argument('--softmax_layer', '-sl', default=False, action="store_true", help='In case of classification branches, use the final layer as softmax')
    parser.add_argument('--classification_loss', '-cl', default='cross-entropy', type=str, help='In case of classification branches, the loss used for the computation')
    parser.add_argument('--iterations', '-i', default=1, type=int, help='Number of models to train')
    parser.add_argument('--loss', '-l', default='smooth_l1', type=str, help='Loss function')
    parser.add_argument('--norm_method', '-nm', default='bn', type=str, help='Normalization method')
    parser.add_argument('--optimizer', '-o', default='adam', type=str, help='Optimizer')
    parser.add_argument('--pool_method', '-pm', default='conv', type=str, help='Pool method')
    parser.add_argument('--pre_train', '-pt', default=False, action='store_true', help='Auto-encoder pre-training')
    parser.add_argument('--retrain', '-r', default='', type=str, help='Model to retrain')
    parser.add_argument('--split', '-s', default='01', type=str, help='Train/val split')

    # Evaluation args - for now just KIT-GE post-processing pipeline args it is implemented
    
    parser.add_argument("--models_split", default="models/trained", type=str, help="Path to fetch the chosen model")
    parser.add_argument("--models_name", default="none", type=str, help="model's name to fetch")
    parser.add_argument('--apply_merging', '-am', default=False, action='store_true', help='Merging post-processing') # merging post-processing (prevents oversegmentation)
    parser.add_argument('--artifact_correction', '-ac', default=False, action='store_true', help='Artifact correction')
    parser.add_argument('--batch_size', '-bs', default=2, type=int, help='Batch size') # NOTE: Originally suggested '8' - used both for val/training methods
    parser.add_argument('--mode', '-m', default='GT', type=str, help='Ground truth type / evaluation mode') # Used both for val/training methods - jsut 'GT' is supported for now
    parser.add_argument('--multi_gpu', '-mgpu', default=False, action='store_true', help='Use multiple GPUs') # Used both for training/validation/test methods
    parser.add_argument('--apply_clahe', '-acl', default=False, action='store_true', help='CLAHE pre-processing')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save some raw predictions')
    parser.add_argument('--scale', '-sc', default=1, type=float, help='Scale factor (0: get from trainset info.json') # json file with per-dataset parameters
    parser.add_argument('--subset', '-ss', default='01', type=str, help='Subset to evaluate on') # Possible options: [01, 02,01+02]
    parser.add_argument('--th_cell', '-tc', default=0.07, nargs='+', help='Threshold for adjusting cell size')
    parser.add_argument('--th_seed', '-ts', default=0.45, nargs='+', help='Threshold for seeds')
    parser.add_argument('--fusion_overlap', '-fo', type=float, default=0.60, help='Percent of overlap of EVs between the features maps')

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

    if not isinstance(args.crop_size, list): # It support multiple crop_size options.
        args.crop_size = [args.crop_size]

    if not isinstance(args.dataset, list):
        args.dataset = [args.dataset]

    # Check args consistency - both True is not supported.
    if args.pre_train and args.retrain:
        raise Exception('Use either the pre-train option --pre_train or the retrain option --retrain')
    return args











