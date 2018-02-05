""" 
Generate 2D bounding box data 
(used as input for 3D bounding box generation)
"""
import os, argparse, glob, imageio
import model as modellib # Mask RCNN
import numpy as np
from unreal_config  import UnrealConfig, InferenceConfig
from unreal_utils   import MODEL_DIR, limit_GPU_usage, compute_mean_AP
from pathlib import Path


def generate_bbox2d(source_image_dir, model_weight_path, instance_dir):
    # limit GPU usage (don't use it all!)
    limit_GPU_usage()

    # prepare the model for inference
    config = InferenceConfig()
    config.display()

    # build the model and the weights
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(model_weight_path, by_name=True)

    class_names = ['BG', 'person', 'car'] # names used by 3D-Deepbox
    for image_path in sorted(glob.glob(os.path.join(source_image_dir, 'lit*.png'))):
        # load the image
        image = imageio.imread(image_path)
        image = image[:, :, :3]
        # run object detection on the image
        results = model.detect([image], verbose=0)
        r = results[0]
        # persist the result (2D bounding boxes)
        fid = os.path.splitext(os.path.basename(image_path))[0]
        npz_path = os.path.join(instance_dir, '{}.npz'.format(fid))
        np.savez_compressed(npz_path, 
                            rois=r['rois'], 
                            class_ids=r['class_ids'], 
                            class_names=class_names, # TODO not very efficient
                            scores=r['scores'], 
                            masks=r['masks'])
        print(image_path, '=>', npz_path)


if __name__=='__main__':
    # constants
    HOME_DIR = str(Path.home())
    SOURCE_IMAGE_PATH = '{}/datasets/unreal/unreal dataset 1/images'.format(HOME_DIR)
    TARGET_DATA_DIR   = '{}/datasets/gtc2018/unreal/unreal dataset 1/instances'.format(HOME_DIR)

    # command line parameters
    parser = argparse.ArgumentParser(description='Unreal Mask RCNN Train')
    parser.add_argument('-s', help='source image dir', dest='source_image_dir',  type=str, default=SOURCE_IMAGE_PATH)
    parser.add_argument('-t', help='target data dir',  dest='target_data_dir',   type=str, default=TARGET_DATA_DIR)
    parser.add_argument('-w', help='weight path',      dest='model_weight_path', type=str, default='unreal_model_weights.h5')
    args = parser.parse_args()
    print(args)

    # generate the 2d bounding box data
    generate_bbox2d(args.source_image_dir, args.model_weight_path, args.target_data_dir)


