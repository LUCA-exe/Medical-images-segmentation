"""main.py

This is a temporary main executable file for running the repository.

Helpful resources:
    - 
"""
from utils import create_logging, download_images
from parser import get_parser, get_processed_args
from img_processing.main_img import *
from download_data import download_datasets


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

  if args.download: # Check if is is requested the donwloading of datasets
    download_datasets(log, args)
  
  # Process single folders signals and aggregate for the dataset
  processor = images_processor(env, args)
  #processor.collect_signals()
  
  visualizator = signalsVisualizator(env, args)
  #visualizator.visualize_signals() # Compute single signals
  signalsVisualizator.dataset_signals_comparison(log) # Compare single signals from different datasets


if __name__ == "__main__":
    main()