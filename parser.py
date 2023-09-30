""" parser.py:

- Configuration of the parameters.
"""

import configargparse

def get_parser():
  """ Standard function to load the parameters

    Returns:
        Object: parser
  """
  parser = configargparse.ArgumentParser(description="Medical images segmentation")

  # Path arguments
  parser.add_argument("--images_path",
                        default="images/",
                        type=str,
                        help="Path to images (both single images and video frame)")

  parser.add_argument("--images_mode", # If required is True why put a 'default' param?
                        default="single",
                        #required=True, 
                        type=str,
                        choices=["single", "time_lapse"],
                        help="Which folder to access")
                        
  return parser


def get_processed_args(args):
  """ Process the args from config file and console to usable format

    Args:
        output (Tensor): predicted probabilities, higher = more confidence

    Returns:
        args
  """
  return args
