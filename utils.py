"""utils.py

- Logger functions 
- Parsing functions
"""

import os
import logging
import datetime

LOGS_PATH = "logs" # Const as the path is fixed

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
  run_log_folder_path = os.path.join(day_log_path, current_time) # Specific folder for a run
  
  info_log_file = f"info_{current_time}.log"
  debug_log_file = f"debug_{current_time}.log"

  # Set up handlers
  info_handler = logging.FileHandler(info_log_file)
  info_handler.setLevel(logging.INFO)
  debug_handler = logging.FileHandler(debug_log_file)
  debug_handler.setLevel(logging.DEBUG)

  # Set up single logger (info and debug)
  logger = logging.getLogger('Logger')
  logger.setLevel(logging.DEBUG) # Lower level for the two handlers

  # Set up the format messages
  log_format = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
  info_handler.setFormatter(log_format)
  debug_handler.setFormatter(log_format)

  # Add the handlers to the loggers
  logger.addHandler(info_handler)
  logger.addHandler(debug_handler)

  return logger