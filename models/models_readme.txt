# Model folder

## Folder Structure
The folder 'Medical-images-segmentaion/models/' contains respectively 4 folders:
- 'trained': Models that are end_to_end trained on a dataset and ready to be evaluated.
- 'pre_trained': Models that are trained on a dataset and waiting to be fine-tuned.
- 'fine_tuned': Models that are fine-tuned after being pre-trained.
- 'all': model to train/fine-tune/evaluate. Default folder to fetch 
         for training/validation/inference; remember to move the desired 
         models to evaluate in this location. (the model in this folder, e.g 
         for trainig, will be automatically saved in the others folders
         respectively)

## Naming convention
The name format of the model files (doens't matter in which folder and both for '*.pth' and '*.json') is:
- <Dataset used for training>_<GT or GT_ST>_<split>_<crop-size>_<pre-processing method>_model_<ID>.pth
- <Dataset used for training>_<GT or GT_ST>_<split>_<crop-size>_<pre-processing method>_model_<ID>.json
    - The architecture config. of the model can be inspected using this file.