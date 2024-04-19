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
from net_utils.utils import load_weights, get_num_workers, save_inference_raw_images, save_inference_final_images, create_model_architecture, save_image, log_final_images_properties


def load_and_get_architecture(log, model_path, device, num_gpus):
    # Load architecture from the "*.json" and prepare it for the evaluation phase.

    # Load model json file to get architecture + filters
    with open('.' + model_path.split('.')[1] + '.json') as f: # Fetch model info from saved '*.json'
        model_settings = json.load(f) # Load model structure

    # Build model
    log.info(f"Bulding model '{model_path}' ..")
    # Create the architecture given the json configuration file.
    log.debug(f"Model settings: {model_settings}")

    # Same util function used in training
    net = create_model_architecture(log, model_settings, device, num_gpus)

    # Load the weights
    load_weights(net, model_path, device, num_gpus)
    
    # Prepare model for evaluation
    net.eval()
    torch.set_grad_enabled(False) # NOTE: This command affect the autograd context-manager just the single thread.
    log.info(f"Model correctly set for evaluation phase")
    return net, model_settings


def move_batches_to_device(samples_dict, device, keys_to_move = ["image", "single_channel_image"]):
    # util function to get the dict. batch from the dataloader and move to the correct device

    # Un-pack the dict.
    for key, tensor in samples_dict.items():
        if key in keys_to_move:

            if not samples_dict[key] is None:
                samples_dict[key] = samples_dict[key].to(device)

    # Return updated dict.
    return samples_dict


# For now use this simple 'dataloader' loop for evaluation of different pipelines.
def inference_2d(log, model_path, data_path, result_path, device, num_gpus, batchsize, args):
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
    net, model_settings = load_and_get_architecture(log, model_path, device, num_gpus)

    # Get images to predict
    ctc_dataset = CTCDataSet(data_dir=data_path,
                             transform=pre_processing_transforms(apply_clahe=args.apply_clahe, scale_factor=args.scale))
    log.info(f"Inference dataset correctly set")

    num_workers = get_num_workers(device)
    num_workers = np.minimum(num_workers, 16)
    log.debug(f"Number of workers set for the dataloader: {num_workers}")

    dataloader = torch.utils.data.DataLoader(ctc_dataset, batch_size=batchsize, shuffle=False, pin_memory=True,
                                             num_workers=num_workers)

    arch_name = model_settings['architecture'][0]
    # Predict images (iterate over images/files)
    for idx, sample in enumerate(dataloader):

        # Pack a dict. for readibility/simplicity
        
        sample = move_batches_to_device(sample, device)
        #img_batch, ids_batch, pad_batch, img_size = sample
        #img_batch = img_batch.to(device)

        # DEBUG
        '''print(sample.keys())
        print(sample["image"].shape)
        save_image(np.squeeze(sample["image"][0].cpu().detach().numpy()[0, :, :]), "./tmp", f"First image")
        save_image(np.squeeze(sample["single_channel_image"][0].cpu().detach().numpy()[0, :, :]), "./tmp", f"First image single channel")
        save_image(np.squeeze(sample["image"][1].cpu().detach().numpy()[0, :, :]), "./tmp", f"Second image")        
        exit(1)'''

        if batchsize > 1:  # all images in a batch have same dimensions and pads
            pad_batch = [sample["pads"][i][0] for i in range(len(sample["pads"]))]
            img_size = [sample['original_size'][i][0] for i in range(len(sample['original_size']))]

        # Prediction outputs - dependent on the loaded model pipeline and the chosen post-processing pipeline to move inside a function in this module

        if arch_name == "dual-unet":

            # NOTE: temporary path for the custom post-processing pipeline
            if args.post_pipeline == "fusion-dual-unet":
                prediction_border_batch, prediction_cell_batch = net(sample["image"])

                # Additional prediction for the single channel images
                sc_prediction_border_batch, sc_prediction_cell_batch = net(sample["single_channel_image"])

                # Adjust the pads and move to cpu for further processing
                prediction_border_batch = prediction_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()

                # Adjust the additional inferred images
                sc_prediction_border_batch = sc_prediction_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
                sc_prediction_cell_batch = sc_prediction_cell_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
            
            else:
                # Normal post-processing pipeline
                prediction_border_batch, prediction_cell_batch = net(sample["image"])
                prediction_border_batch = prediction_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()

        elif arch_name == "original-dual-unet":
            prediction_binary_border_batch, prediction_cell_batch, prediction_mask_batch = net(sample["image"])
            # NOTE: The selection of the "padded" dim has to comprhend all the channels in case of binary prediction
            prediction_mask_batch = prediction_mask_batch[:, :, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
            prediction_binary_border_batch = prediction_binary_border_batch[:, :, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
            #prediction_mask_batch = prediction_mask_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
            #prediction_binary_border_batch = prediction_binary_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()

        elif arch_name == "triple-unet":
            prediction_border_batch, prediction_cell_batch, prediction_mask_batch = net(sample["image"])
            prediction_mask_batch = prediction_mask_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
            prediction_border_batch = prediction_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()

        log.debug(f".. predicted batch {idx} ..")

        # Get rid of pads
        prediction_cell_batch = prediction_cell_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()
        #prediction_border_batch = prediction_border_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()

        # Save also some raw predictions (not all since float32 --> needs lot of memory)
        save_ids = [0, len(ctc_dataset) // 8, len(ctc_dataset) // 4, 3 * len(ctc_dataset) // 8, len(ctc_dataset) // 2,
                    5 * len(ctc_dataset), 3 * len(ctc_dataset) // 4, 7 * len(ctc_dataset) // 8, len(ctc_dataset) - 1]

        # Go through predicted batch and apply post-processing (not parallelized)
        for h in range(len(prediction_cell_batch)): # NOTE: For now this batch is the only one that is always computed - to change!

            log.debug('.. processing {0} ..'.format(sample["id"][h]))

            # Get actual file number:
            file_num = int(sample["id"][h].split('t')[-1])

            # Save not all raw predictions to save memory
            if file_num in save_ids and args.save_raw_pred:
                save_raw_pred = True
            else:
                save_raw_pred = False

            file_id = sample["id"][h].split('t')[-1] + '.tif'

            # Implementing different post-processing options
            if args.post_pipeline == 'dual-unet':
                
                prediction_instance, border = border_cell_post_processing(border_prediction=prediction_border_batch[h],
                                                                    cell_prediction=prediction_cell_batch[h],
                                                                    args=args)

            if args.post_pipeline == 'fusion-dual-unet':
                
                prediction_instance, border = sc_border_cell_post_processing(border_prediction=prediction_border_batch[h],
                                                                    cell_prediction=prediction_cell_batch[h],
                                                                    sc_border_prediction=sc_prediction_border_batch[h],
                                                                    sc_cell_prediction=sc_prediction_cell_batch[h],
                                                                    args=args)
            
            # TO FINISH TEST
            if args.post_pipeline == 'triple-unet':

                prediction_instance = seg_mask_post_processing(mask = prediction_mask_batch[h], args = args)
            

            if args.post_pipeline == 'original-dual-unet':
                
                prediction_instance = simple_binary_mask_post_processing(mask = prediction_mask_batch[h], original_image = sample["image"][h].cpu().numpy(), args = args)
                #prediction_instance = complex_binary_mask_post_processing(mask = prediction_mask_batch[h], binary_border=prediction_binary_border_batch[h], cell_prediction=prediction_cell_batch[h], original_image = sample["image"][h].cpu().numpy(), args = args)
            
            if args.scale < 1:
                prediction_instance = resize(prediction_instance,
                                             img_size,
                                             order=0,
                                             preserve_range=True,
                                             anti_aliasing=False).astype(np.uint16)

            prediction_instance = foi_correction(mask=prediction_instance, cell_type=args.cell_type)

            # Log the caracteristics of the saved images with an util functions
            log_final_images_properties(log, prediction_instance)
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