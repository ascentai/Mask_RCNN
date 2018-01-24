"""
Commonly used utility functions
"""
import os
import tensorflow as tf
import numpy as np
from keras import backend as K
import model as modellib # Mask RCNN
import utils


# constatns
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


def limit_GPU_usage():
    """ Prevent tensorflow from using up all GPUs
    """
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=cfg))


def load_weights(model, init_with):
    """
    """
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
        print('loaded imagenet weights')
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        model.load_weights(COCO_MODEL_PATH, by_name=True,
           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        print('loaded coco weights')
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
        print('loaded last trained wegihts')


def compute_mean_AP(model, config, dataset, n_images):
    """ Compute VOC-Style mAP @ IoU=0.5        
    """
    image_ids = np.random.choice(dataset.image_ids, n_images)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        result = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        if len(result)==5:
            image, image_meta, class_ids, gt_bbox, gt_mask = result
        else:
            image, image_meta, gt_bbox, gt_mask = result
            class_ids = gt_bbox[:,4]
            gt_bbox = gt_bbox[:, :4]
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, class_ids,
                             r["rois"], r["class_ids"], r["scores"])
        APs.append(AP)
    return np.mean(APs)

