import os, argparse
import model as modellib # Mask RCNN
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
    #TODO test set should be different from the validation set
    OBJECT_DESC_PATH = '{}/datasets/unreal/unreal dataset 2/objects_description.json'.format(HOME_DIR)
    IMAGES_PATH      = '{}/datasets/unreal/unreal dataset 2/images'.format(HOME_DIR)

    parser = argparse.ArgumentParser(description='Unreal Mask RCNN Train')
    parser.add_argument('-o', help='object path', dest='test_object_path',  type=str, default=OBJECT_DESC_PATH)
    parser.add_argument('-i', help='images path', dest='test_images_path',  type=str, default=IMAGES_PATH)
    parser.add_argument('-w', help='weight path', dest='model_weight_path', type=str, default='unreal_model_weights.h5')
    parser.add_argument('-n', help='image count', dest='image_count',       type=int, default=10)

    args = parser.parse_args() 

    # Test dataset
    dataset_test = UnrealDataset()
    dataset_test.populate(args.test_object_path, args.test_images_path)
    dataset_test.prepare()

    # limit GPU usage (don't use it all!)
    limit_GPU_usage()

    # prepare the model for inference
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(args.model_weight_path, by_name=True)

    # calculate the mean average precision
    mean_AP = compute_mean_AP(model, config, dataset_test, args.image_count)
    print('Mean AP:', mean_AP)

