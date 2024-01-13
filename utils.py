"""utils.py

- Logger functions 
- Parsing functions
- Containing specific class args parsing
- Set the device for train/eval
"""

import torch
import os
import logging
from datetime import datetime
import json

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


# TODO: this class offer a customized EvaluationParser for every implemented pipeline.
class EvalArgs(object): # Class containings inference and post-processing parameters.
    """ Class with post-processing parameters.
    """

    def __init__(self, post_processing_pipeline, th_cell, th_seed, apply_clahe, scale, cell_type,
                 save_raw_pred,artifact_correction, apply_merging):
        """
        (kit-ge post-processing params)
        :param th_cell: Mask / cell size threshold.
            :type th_cell: float
        :param th_seed: Seed / marker threshold.
            :type th_seed: float
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
        """
        if post_processing_pipeline == 'kit-ge':
            self.th_cell = th_cell
            self.th_seed = th_seed
            self.apply_clahe = apply_clahe
            self.scale = scale
            self.cell_type = cell_type
            self.save_raw_pred = save_raw_pred
            self.artifact_correction = artifact_correction
            self.apply_merging = apply_merging

    # Override default class function to print parameters
    def __str__(self):
        attributes = ', '.join(f'{key}={value}' for key, value in vars(self).items())
        return f"EvalArgs({attributes})"


# TODO: this class offer a customized TrainingParser for every implemented pipeline.
class TrainArgs(object):
    """ Class with training creation parameters.
    """

    def __init__(self, model_pipeline, act_fun, batch_size, filters, iterations,
    loss, norm_method, optimizer, pool_method, pre_train, retrain, split):
        """ kit-ge training params implemented for now.
        """
        if model_pipeline == 'kit-ge':
            self.act_fun = act_fun
            self.batch_size = batch_size
            self.filters = filters
            self.iterations = iterations
            self.loss = loss
            self.norm_method = norm_method
            self.optimizer = optimizer
            self.pool_method = pool_method
            self.pre_train = pre_train
            self.retrain = retrain
            self.split = split

    # Override default class function
    def __str__(self):
        attributes = ', '.join(f'{key}={value}' for key, value in vars(self).items())
        return f"TrainArgs({attributes})"


        