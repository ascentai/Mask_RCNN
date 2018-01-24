"""
Compute mean AP (mAP) to evaluate the performance of object detection
"""
import os, argparse
import model as modellib # Mask RCNN
from unreal_config  import UnrealConfig, InferenceConfig
from unreal_dataset import UnrealDataset
from unreal_utils   import MODEL_DIR, limit_GPU_usage, compute_mean_AP
from pathlib import Path


def evaluate_mAP(source_image_dir, object_desc_path, model_weight_path, image_count):
    # load dataset
    dataset = UnrealDataset()
    dataset.populate(source_image_dir, object_desc_path)
    dataset.prepare()

    # limit GPU usage (don't use it all!)
    limit_GPU_usage()

    # prepare the model for inference
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(model_weight_path, by_name=True)

    # calculate the mean average precision
    mean_AP = compute_mean_AP(model, config, dataset, image_count)
    print('Mean AP:', mean_AP)

if __name__=='__main__':
    HOME_DIR = str(Path.home())
    #TODO test set should be different from the validation set
    SOURCE_IMAGE_DIR = '{}/datasets/unreal/unreal dataset 1/images'.format(HOME_DIR)
    OBJECT_DESC_PATH = '{}/datasets/unreal/unreal dataset 1/objects_description.json'.format(HOME_DIR)

    parser = argparse.ArgumentParser(description='Unreal Mask RCNN Train')
    parser.add_argument('-s', help='source image dir',  dest='source_image_dir',  type=str, default=SOURCE_IMAGE_DIR)
    parser.add_argument('-o', help='object desc path',  dest='object_desc_path',  type=str, default=OBJECT_DESC_PATH)
    parser.add_argument('-w', help='model weight path', dest='model_weight_path', type=str, default='unreal_model_weights.h5')
    parser.add_argument('-n', help='image count',       dest='image_count',       type=int, default=10)
    args = parser.parse_args()
    print(args)

    evaluate_mAP(args.source_image_dir, args.object_desc_path, args.model_weight_path, args.image_count)

