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
    else:
        model.load_weights(init_with, by_name=True)
        print('loaded the weights in ' + init_with)


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

def compute_mean_AP_from_annotations(class_2_num, annot_detect_file, annot_gt_file): #(annot_gt_file, annot_detect_file):
    #annot_detect_file = '/nas/datashare/datasets/elecparts/3/results/epoch_009/correct_rotation_and_our_resize/result_real.pkl'
    #annot_gt_file = '/nas/datashare/datasets/elecparts/3/12objects_real/annotation.pkl'
    # from elecparts_dataset import elecpartsDataset
    # dataset = elecpartsDataset()
    # class_2_num = dict(zip(dataset.my_class_names, np.arange(len(dataset.my_class_names))))
    import pickle
    APs = []
    annot_gt = pickle.load( open(annot_gt_file,'rb'))
    annot_detect = pickle.load( open(annot_detect_file,'rb'))

    ks = annot_gt.keys()
    for k in ks:
        # find the detection for the current image
      	found = False
	for detect in annot_detect:
		if detect['filename'] == annot_gt[k]['filename']:
			found = True
			detect_for_current_image = detect
			break
	assert found
        
        # prepare the bboxes, class_ids and scores from the annotation and detect for the current image
        gt_bboxes = []
        gt_class_ids = []
        bboxes = []
        class_ids = []
        scores = []
        for a in annot_gt[k]['annot']:
            bbox = a['bbox']
            gt_bboxes.append([bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]])
            gt_class_ids.append(class_2_num[a['label']])
        for a in detect_for_current_image:
            bbox = a['bbox']
            bboxes.append([bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]])
            class_ids.append(class_2_num[a['label']])
            scores.append(a['score'])

                
        AP, precisions, recalls, overlaps = utils.compute_ap(np.array(gt_bboxes), np.array(gt_class_ids), np.array(bboxes), np.array(class_ids), np.array(scores))

        APs.append(AP)
	
    return np.mean(APs), APs

