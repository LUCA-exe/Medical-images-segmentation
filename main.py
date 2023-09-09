"""main.py

This is the main executable file for running the Time series analysis code.
Processing and analysis of time series and  Training/validation/testing of the models occurs from this entrypoint.

Helpful resources:
    - 
"""

from parser import get_parser, get_processed_args

def main():
  """ Main function to call in order to run all the project classes and functions
  """
  args = get_parser().parse_args()
  args = get_processed_args(args)

  print("args: \n")
  pprint.pprint(args)

if __name__ == "__main__":
    main()

