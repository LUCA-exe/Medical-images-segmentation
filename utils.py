"""utils.py

- Logger functions 
- Parsing functions
"""

import os

def set_up_info_logging():
  """Function to set up the high-level logging
  """ 
  log_folder_path = Path("logs")
  if not log_folder_path.exists():
    log_folder_path.mkdir(parents=True)
    print("Logs folder created.")
  else:
    print("Logs folder already exists.")

  # Get the current date and time
  current_datetime = datetime.now()
  # Extract the date, hour, and minute components as strings
  current_date = current_datetime.strftime("%Y-%m-%d")
  current_hour = current_datetime.strftime("%H")
  current_minute = current_datetime.strftime("%M")
  log_file_name = f"log_{current_date}_{current_hour}_{current_minute}.log"

  # Create the file path
  file_path = Path(log_folder_path) / log_file_name
  file_path_str = str(file_path)

  # Save the file
  try:
    logging.basicConfig(filename=file_path_str, filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
  except AssertionError as error:
    print(f"Can't create log file >>> Error: {error}")
  
  return file_path_str # Return to pass the file to the Trainer class