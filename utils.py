"""utils.py

- Logger functions 
- Parsing functions
"""

import torch
import os
import logging
from datetime import datetime

LOGS_PATH = "logs" # TODO: Const as the path is fixed. Move this in a 'ENV' dict 

def create_logging():
  """ Function to set up the 'INFO' and 'DEBUG' log file
  """ 
  os.makedirs(LOGS_PATH, exist_ok=True)

  # Get the current date and time
  current_datetime = datetime.now()

  # Extract the date and  hour, and minute components for the sub-folder
  current_date = current_datetime.strftime("%Y-%m-%d") 
  day_log_path = os.path.join(LOGS_PATH, current_date) # Folder for the day's logs
  os.makedirs(day_log_path, exist_ok=True)

  current_time = current_datetime.strftime("%H-%M-%S") 
  run_log_path = os.path.join(day_log_path, current_time) # Specific folder for a run
  os.makedirs(run_log_path, exist_ok=True)
  
  info_log_file = f"info_{current_time}.log"
  debug_log_file = f"debug_{current_time}.log"
  
  # Complete paths for the logs file
  info_log_path = os.path.join(run_log_path, info_log_file)
  debug_log_path = os.path.join(run_log_path, debug_log_file)
  
  # Set up handlers
  info_handler = logging.FileHandler(info_log_path)
  info_handler.setLevel(logging.INFO)
  debug_handler = logging.FileHandler(debug_log_path)
  debug_handler.setLevel(logging.DEBUG)

  # Set up single logger (info and debug)
  logger = logging.getLogger('logger') # Name will remain fixed 
  logger.setLevel(logging.DEBUG) # Lower level for the two handlers

  # Set up the format messages
  log_info_format = logging.Formatter('%(asctime)s - %(levelname)s   %(message)s')
  log_debug_format = logging.Formatter('%(asctime)s - %(levelname)s  %(message)s')
  info_handler.setFormatter(log_info_format)
  debug_handler.setFormatter(log_debug_format)

  # Add the handlers to the loggers
  logger.addHandler(info_handler)
  logger.addHandler(debug_handler)

  return logger


# For now the repository is implemented for single-gpu usage.
def set_device():
    # Set device for using CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
        num_gpus = 1

    return device


# TODO: Check compared to the original repository
class EvalArgs(object): # Class containings inference and post-processing parameters
    """ Class with post-processing parameters.

    """

    def __init__(self, th_cell, th_seed, apply_clahe, scale, cell_type,
                 artifact_correction, apply_merging):
        """

        :param th_cell: Mask / cell size threshold.
            :type th_cell: float
        :param th_seed: Seed / marker threshold.
            :type th_seed: float
        :param n_splitting: Number of detected cells above which to apply additional splitting (only for 3D).
            :type n_splitting: int
        :param apply_clahe: Apply contrast limited adaptive histogram equalization (CLAHE).
            :type apply_clahe: bool
        :param scale: Scale factor for downsampling.
            :type scale: float
        :param cell_type: Cell type.
            :type cell_type: str
        :param save_raw_pred: Save (some) raw predictions.
            :type save_raw_pred: bool
        :param artifact_correction: Apply artifact correction post-processing.
            :type artifact_correction: bool
        :param fuse_z_seeds: Fuse seeds in z-direction / axial direction.
            :type fuse_z_seeds: bool
        """
        self.th_cell = th_cell
        self.th_seed = th_seed
        self.apply_clahe = apply_clahe
        self.scale = scale
        self.artifact_correction = artifact_correction
        self.apply_merging = apply_merging


        