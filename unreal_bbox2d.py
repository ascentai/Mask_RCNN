import os, argparse
import model as modellib # Mask RCNN
import numpy as np
from unreal_config  import UnrealConfig
from unreal_dataset import UnrealDataset
from unreal_utils   import MODEL_DIR, limit_GPU_usage, compute_mean_AP
from pathlib import Path


class InferenceConfig(UnrealConfig):
    """ config for testing / prediciton / inference
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__=='__main__':
    HOME_DIR = str(Path.home())
    OBJECT_DESC_PATH = '{}/datasets/unreal/unreal dataset 1/objects_description.json'.format(HOME_DIR)
    IMAGES_PATH      = '{}/datasets/unreal/unreal dataset 1/images'.format(HOME_DIR)
    DATA_PATH        = '{}/datasets/unreal/unreal dataset 1/data'.format(HOME_DIR)

    parser = argparse.ArgumentParser(description='Unreal Mask RCNN Train')
    parser.add_argument('-o', help='object path', dest='object_path',       type=str, default=OBJECT_DESC_PATH)
    parser.add_argument('-i', help='images path', dest='images_path',       type=str, default=IMAGES_PATH)
    parser.add_argument('-d', help='data path',   dest='data_path',         type=str, default=DATA_PATH)
    parser.add_argument('-w', help='weight path', dest='model_weight_path', type=str, default='unreal_model_weights.h5')

    args = parser.parse_args() 

    # Test dataset
    dataset = UnrealDataset()
    dataset.populate(args.object_path, args.images_path)
    dataset.prepare()

    # limit GPU usage (don't use it all!)
    limit_GPU_usage()

    # prepare the model for inference
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(args.model_weight_path, by_name=True)

    class_names = ['BG', 'people', 'car'] # names used by 3D-Deepbox
    for image_id in sorted(dataset.image_ids):
        # Load image and ground truth data
        result = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        if len(result)==5:
            image, image_meta, class_ids, gt_bbox, gt_mask = result
        else:
            image, image_meta, gt_bbox, gt_mask = result
            class_ids = gt_bbox[:,4]
            gt_bbox = gt_bbox[:, :4]
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        image_path = dataset.image_reference(image_id)
        npz_file = '{}{}{}'.format(
            args.data_path, os.sep, image_path.split(os.sep)[-1].replace('.png', '.npz'))
        np.savez_compressed(npz_file, 
                            rois=r['rois'], 
                            class_ids=r['class_ids'], 
                            class_names=class_names, # TODO not very efficient
                            scores=r['scores'], 
                            masks=r['masks'])
        print(image_id, image_path, npz_file)
