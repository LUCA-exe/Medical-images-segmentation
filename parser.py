"""
parser.py:

- Configuration of the parameters.
"""

import configargparse

def get_parser():
  """ Standard function to load the parameters

    Returns:
        Object: parser
  """
  parser = configargparse.ArgumentParser(description="Time series analysis parser")
  return parser


def get_processed_args(args):
  """ Process the args from config file and console to usable format

    Args:
        output (Tensor): predicted probabilities, higher = more confidence

    Returns:
        args
  """
  return args
