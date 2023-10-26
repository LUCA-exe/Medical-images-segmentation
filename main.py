"""main.py

This is a temporary main executable file for running the repository.

Helpful resources:
    - 
"""
from utils import create_logging, download_images
from parser import get_parser, get_processed_args
from img_processing.main_img import *


def main():
  """ Main function to call in order to run all the project classes and functions
  """
  log = create_logging() # Set up 'logger' object 

  args = get_parser() # Set up dict arguments
  args = get_processed_args(args)
  
  env = {} # TODO: Load this from a '.json' file
  env['logger'] = log # Move the object through 'env' dict

  log.info(f"args: {args}") # Print overall args 
  log.debug(f"env: {env}")

  download_images(env, args) # Set up the images folder
  
  processor = images_processor(env, args)
  processor.collect_signals()


if __name__ == "__main__":
    main()