"""
Train Mask-RCNN on Unreal synthetic images
"""
import os, argparse
import model as modellib # Mask RCNN
from unreal_config  import UnrealConfig
from unreal_dataset import UnrealDataset
from unreal_utils   import MODEL_DIR, limit_GPU_usage, load_weights
from pathlib import Path


def train_model(
    train_source_image_dir,
    train_object_desc_path,
    valid_source_image_dir,
    valid_object_desc_path,
    init_with,
    epochs,
    epochs2,
    model_weight_path):
    # Training dataset
    dataset_train = UnrealDataset()
    dataset_train.populate(train_source_image_dir, train_object_desc_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_valid = UnrealDataset()
    dataset_valid.populate(valid_source_image_dir, valid_object_desc_path)
    dataset_valid.prepare()

    # don't use all GPUs!
    limit_GPU_usage()

    # prepare the model for training
    config = UnrealConfig()
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    load_weights(model, init_with)

    # train the head branches: passing layers="heads" freezes all layers except the head layers.
    model.train(dataset_train, dataset_valid, 
        learning_rate=config.LEARNING_RATE, 
        epochs=epochs,
        layers='heads')

    # fine tuning
    model.train(dataset_train, dataset_valid, 
        learning_rate=config.LEARNING_RATE/10,
        epochs=epochs2,
        layers='all')

    model.keras_model.save_weights(model_weight_path)
    print('saved {}'.format(model_weight_path))


if __name__=='__main__':
    # constants
    HOME_DIR = str(Path.home())
    TRAIN_SOURCE_IMAGE_DIR = '{}/datasets/unreal/unreal dataset 3/images'.format(HOME_DIR)
    TRAIN_OBJECT_DESC_PATH = '{}/datasets/unreal/unreal dataset 3/objects_description.json'.format(HOME_DIR)
    VALID_SOURCE_IMAGE_DIR = '{}/datasets/unreal/unreal dataset 1/images'.format(HOME_DIR)
    VALID_OBJECT_DESC_PATH = '{}/datasets/unreal/unreal dataset 1/objects_description.json'.format(HOME_DIR)

    # command line parameters
    parser = argparse.ArgumentParser(description='Unreal Mask RCNN Train')
    parser.add_argument('-s',  help='train source image dir', dest='train_source_image_dir', type=str, default=TRAIN_SOURCE_IMAGE_DIR)
    parser.add_argument('-o',  help='train object desc path', dest='train_object_desc_path', type=str, default=TRAIN_OBJECT_DESC_PATH)
    parser.add_argument('-s2', help='valid source image dir', dest='valid_source_image_dir', type=str, default=VALID_SOURCE_IMAGE_DIR)
    parser.add_argument('-o2', help='valid object desc path', dest='valid_object_desc_path', type=str, default=VALID_OBJECT_DESC_PATH)
    parser.add_argument('-k',  help='initial weight type',    dest='init_with',              type=str, default='coco')
    parser.add_argument('-w',  help='weight path',            dest='model_weight_path',      type=str, default='unreal_model_weights.h5')
    parser.add_argument('-e',  help='training epochs',        dest='epochs',                 type=int, default='10')
    parser.add_argument('-e2', help='fine tune epochs',       dest='epochs2',                type=int, default='2')
    args = parser.parse_args() 
    print(args)

    # train the model
    train_model(
        args.train_source_image_dir,
        args.train_object_desc_path,
        args.valid_source_image_dir,
        args.valid_object_desc_path,
        args.init_with,
        args.epochs,
        args.epochs2,
        args.model_weight_path)
