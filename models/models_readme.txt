The folder './models/' contains respectively 4 folders:
- 'trained': Models that are end_to_end trained on a dataset and ready to be evaluated.
- 'pre-trained': Models that are trained on a dataset and waiting to be fine-tuned.
- 'fine-tuned': Models that are fine-tuned after being pre-trained.
- 'all': model to train/fine-tune/evaluate. Default folder to fetch 
         for training/validation/inference; remember to move the desired 
         models to evaluate in this location.

The name format of the model files (doens't matter in which folder and both for '*.pth' and '*.json') is:
1. <Dataset used for training>_<GT or GT_ST>_model.pth
2. <Dataset used for training>_<GT or GT_ST>_<Dataset used for fine-tuning>_<GT or GT_ST>_model.pth

The specific architecture of the model can be inspected on the 'log.DEBUG' file and in the 
'metrics.json' file with the fixed and variables args used for the model's evaluation.

NOTE: In case of both training/fine-tuning are used all the GT_<id> experiments of the dataset.
