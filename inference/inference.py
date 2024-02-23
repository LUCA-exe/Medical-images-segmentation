'''This file is used for main inference loop: depending on the chosen model and
post-processing method that can be chosen by parameters'''

import gc
import json
import tifffile as tiff
import torch
import numpy as np
from pathlib import Path

from multiprocessing import cpu_count
from skimage.measure import regionprops, label
from skimage.transform import resize

from inference.ctc_dataset import CTCDataSet, pre_processing_transforms
from inference.postprocessing import *
from net_utils.unets import build_unet
from net_utils.utils import load_weights, get_num_workers, save_inference_raw_images, save_inference_final_images


def create_architecture(log, model_pipeline, model_settings, device, num_gpus):
    # TODO: Check which model to build (implement different pipelines/options to build the model)
    
    if model_pipeline == 'kit-ge':
        net = build_unet(log, unet_type=model_settings['architecture'][0],
                        act_fun=model_settings['architecture'][2],
                        pool_method=model_settings['architecture'][1],
                        normalization=model_settings['architecture'][3],
                        device=device,
                        num_gpus=num_gpus,
                        ch_in=1,
                        ch_out=1,
                        filters=model_settings['architecture'][4],
                        detach_fusion_layers=model_settings['architecture'][5])
    return net


def load_and_get_architecture(log, model_path, model_pipeline, device, num_gpus):
    # Load architecture from the "*.json" and prepare it for the evaluation phase.

    # Load model json file to get architecture + filters
    with open('.' + model_path.split('.')[1] + '.json') as f: # Fetch model info from saved '*.json'
        model_settings = json.load(f) # Load model structure

    # Build model
    log.info(f"Bulding model '{model_path}' ..")
    # Create the architecture given the json configuration file.
    net = create_architecture(log, model_pipeline, model_settings, device, num_gpus)
    # Load the weights.
    load_weights(net, model_path, device, num_gpus)
    
    # Prepare model for evaluation
    net.eval()
    torch.set_grad_enabled(False) # NOTE: This command affect the autograd context-manager just the single thread.
    log.info(f"Model correctly set for evaluation phase")
    return net, model_settings


# For now use this simple 'dataloader' loop for evaluation of different pipelines.
def inference_2d(log, model_path, data_path, result_path, device, num_gpus, batchsize, args, model_pipeline='kit-ge', post_processing_pipeline='kit-ge'):
    """ Inference function for 2D Cell Tracking Challenge data sets.

    :param model: Path to the model to use for inference.
        :type model: pathlib Path object.
    :param data_path: Path to the directory containing the Cell Tracking Challenge data sets.
        :type data_path: pathlib Path object
    :param result_path: Path to the results directory.
        :type result_path: pathlib Path object
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param batchsize: Batch size.
        :type batchsize: int
    :param args: Arguments for post-processing.
        :type args:
    :param num_gpus: Number of GPUs to use in GPU mode (enables larger batches)
        :type num_gpus: int
    :return: None
    """
    result_path = Path(result_path) # Cast from str to Path type.
    net, model_settings = load_and_get_architecture(log, model_path, model_pipeline, device, num_gpus)

    # Get images to predict
    ctc_dataset = CTCDataSet(data_dir=data_path,
                             transform=pre_processing_transforms(apply_clahe=args.apply_clahe, scale_factor=args.scale))
    log.info(f"Inference dataset correctly set")

    num_workers = get_num_workers(device)
    num_workers = np.minimum(num_workers, 16)
    log.debug(f"Number of workers set for the dataloader: {num_workers}")

    dataloader = torch.utils.data.DataLoader(ctc_dataset, batch_size=batchsize, shuffle=False, pin_memory=True,
                                             num_workers=num_workers)

    # Predict images (iterate over images/files)
    for idx, sample in enumerate(dataloader):

        img_batch, ids_batch, pad_batch, img_size = sample
        img_batch = img_batch.to(device)

        if batchsize > 1:  # all images in a batch have same dimensions and pads
            pad_batch = [pad_batch[i][0] for i in range(len(pad_batch))]
            img_size = [img_size[i][0] for i in range(len(img_size))]

        # Prediction outputs - dependent on the chosen model pipeline. 
        if model_pipeline == 'kit-ge':
            prediction_border_batch, prediction_cell_batch = net(img_batch)

        log.debug(f".. predicted batch {idx} ..")

        # Get rid of pads.
        prediction_cell_batch = prediction_cell_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
        prediction_border_batch = prediction_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()

        # Save also some raw predictions (not all since float32 --> needs lot of memory)
        save_ids = [0, len(ctc_dataset) // 8, len(ctc_dataset) // 4, 3 * len(ctc_dataset) // 8, len(ctc_dataset) // 2,
                    5 * len(ctc_dataset), 3 * len(ctc_dataset) // 4, 7 * len(ctc_dataset) // 8, len(ctc_dataset) - 1]

        # Go through predicted batch and apply post-processing (not parallelized)
        for h in range(len(prediction_border_batch)):

            log.debug('.. processing {0} ..'.format(ids_batch[h]))

            # Get actual file number:
            file_num = int(ids_batch[h].split('t')[-1])

            # Save not all raw predictions to save memory
            if file_num in save_ids and args.save_raw_pred:
                save_raw_pred = True
            else:
                save_raw_pred = False

            file_id = ids_batch[h].split('t')[-1] + '.tif'

            # TODO: Implement different post-processing options.
            if post_processing_pipeline == 'kit-ge':
                prediction_instance, border = distance_postprocessing(border_prediction=prediction_border_batch[h],
                                                                    cell_prediction=prediction_cell_batch[h],
                                                                    args=args)


            if args.scale < 1:
                prediction_instance = resize(prediction_instance,
                                             img_size,
                                             order=0,
                                             preserve_range=True,
                                             anti_aliasing=False).astype(np.uint16)

            prediction_instance = foi_correction(mask=prediction_instance, cell_type=args.cell_type)

            # Save images in the 'results' folder
            save_inference_final_images(result_path, file_id, prediction_instance)
            if save_raw_pred: save_inference_raw_images(result_path, file_id, prediction_cell_batch, prediction_border_batch, border)

    # TODO: Move into "post_processing.py" module.
    if args.artifact_correction:
        # Artifact correction based on the assumption that the cells are dense and artifacts far away
        roi = np.zeros_like(prediction_instance) > 0
        prediction_instance_ids = sorted(result_path.glob('mask*'))
        for prediction_instance_id in prediction_instance_ids:
            roi = roi | (tiff.imread(str(prediction_instance_id)) > 0)
        roi = binary_dilation(roi, np.ones(shape=(20, 20)))
        roi = label(roi)
        props = regionprops(roi)
        # Keep only the largest region
        largest_area, largest_area_id = 0, 0
        for prop in props:
            if prop.area > largest_area:
                largest_area = prop.area
                largest_area_id = prop.label
        roi = (roi == largest_area_id)
        for prediction_instance_id in prediction_instance_ids:
            prediction_instance = tiff.imread(str(prediction_instance_id))
            prediction_instance = prediction_instance * roi
            tiff.imwrite(str(prediction_instance_id), prediction_instance.astype(np.uint16))

    # Clear memory
    del net
    gc.collect()

    return model_settings # Return this params to save in the results dict