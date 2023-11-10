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

    # Path arguments
    parser.add_argument("--train_images_path",
                            default="training_data/",
                            type=str,
                            help="Path to the train images dataset")

    parser.add_argument("--test_images_path",
                            default="test_data/",
                            type=str,
                            help="Path to the test images dataset")

    parser.add_argument("--dataset", # If required is True why put a 'default' param?
                            default="Fluo-E2DV-train",
                            type=str,
                            #choices=["single-frame", "multiple_frames"],
                            help="Which folder to access")

    parser.add_argument("--download",
                            default=False,
                            type=bool,
                            help="Boolean value to check for download online datasets")

    # TODO: Expand the dataset options from other challenges/websites
    # What dataset to download (from CTC website)
    parser.add_argument("--download_dataset",
                            default = "all",
                            type=str,
                            help="Name of the dataset to download (else 'all')")

    # WARNING: Usless in trial phase.
    parser.add_argument("--link_images", # TODO: Move this to a 'config.json' or other options
                                        # For now download from my private google drive.. this arg. is temporary unused
                            required=True, 
                            type=str,
                            help="Link to download the dataset")
    
    # Starting args used for the image characteristics gathering

    # TODO: Kept fixed as arguments.. maybe should be extracted from a '*.json' file with a finetuned dimension for every seen dataset..
    parser.add_argument("--cell_dim", # If required is True why put a 'default' param?
                            default = 7000, # My dataset has an avg of 4000 EVs dim in pixels 
                            type=int,
                            help="Min. dimension (in pixels) to consider for gathering cells stats/signals")

    parser.add_argument("--max_images", 
                            default = 25,
                            type=int,
                            help="Max. number of images to take for the gathering of signals (for every folder separately)")
                            
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
