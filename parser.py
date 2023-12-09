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
                            
    args = parser.parse_args()
    return args


def get_processed_args(args):
  """ Process the args from config file and console to usable format

    Args:
        output (Tensor): predicted probabilities, higher = more confidence

    Returns:
        args (dict):
  """
  return args
