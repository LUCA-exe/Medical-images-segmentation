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
  parser.add_argument("--images_path",
                        default="images/",
                        type=str,
                        help="Path to donwload images (both single images and video frame)")

  parser.add_argument("--images_mode", # If required is True why put a 'default' param?
                        default="single",
                        #required=True, 
                        type=str,
                        choices=["single", "time_lapse"],
                        help="Which folder to access")

  # WARNING: Usless in trial phase.
  parser.add_argument("--link_images", # TODO: Move this to a 'config.json' or other options
                                       # For now download from my private google drive.. this arg. unused
                        required=True, 
                        type=str,
                        help="Link to download the dataset")
                        
  return parser


def get_processed_args(args):
  """ Process the args from config file and console to usable format

    Args:
        output (Tensor): predicted probabilities, higher = more confidence

    Returns:
        args (dict):
  """
  return args
