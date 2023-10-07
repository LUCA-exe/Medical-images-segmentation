"""main_img.py

This is the main executable file for running the processing of images functions.
"""

class images_processor:

  def __init__(self, env, args):

    self.env = env # Load overall dicts to be used along the class
    self.args = args


  def compute_signals(self, split='all'):
    """ Load the images from a single split or multiple and compute for every one Its properties

    Args:
        split (str): What split to take in order to compute the images

    Returns:
        None 
  """

  # Util functions

  def __load_image__(image_folder): # Util function of this class. Check if this pattern make sense
    """ Load the single channel or multiple channels of the required image.
        It return a dict with id and all the different images for each 'version' (only nuclei, only boundaries etc ..)

    Args:
        image_folder (str): Path to the different version of the image

    Returns:
        dict: {'id': image_folder, 'dapi': Image object, 'fitc': image object .. }
  """
