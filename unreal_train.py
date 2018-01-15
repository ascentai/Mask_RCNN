import os, argparse
import model as modellib # Mask RCNN
from unreal_config  import UnrealConfig
from unreal_dataset import UnrealDataset
from unreal_utils   import MODEL_DIR, limit_GPU_usage, load_weights
from pathlib import Path


if __name__=='__main__':
    HOME_DIR = str(Path.home())
    OBJECT_PATH1 = '{}/datasets/unreal/unreal dataset 1/objects_description.json'.format(HOME_DIR)
    IMAGES_PATH1 = '{}/datasets/unreal/unreal dataset 1/images'.format(HOME_DIR)
    OBJECT_PATH2 = '{}/datasets/unreal/unreal dataset 2/objects_description.json'.format(HOME_DIR)
    IMAGES_PATH2 = '{}/datasets/unreal/unreal dataset 2/images'.format(HOME_DIR)

    parser = argparse.ArgumentParser(description='Unreal Mask RCNN Train')
    parser.add_argument('-o',  help='object path',              dest='train_object_path', type=str, default=OBJECT_PATH1)
    parser.add_argument('-i',  help='images path',              dest='train_images_path', type=str, default=IMAGES_PATH1)
    parser.add_argument('-o2', help='object path (validation)', dest='valid_object_path', type=str, default=OBJECT_PATH2)
    parser.add_argument('-i2', help='images path (validation)', dest='valid_images_path', type=str, default=IMAGES_PATH2)
    parser.add_argument('-k',  help='initial weight type',      dest='init_with',         type=str, default='coco')
    parser.add_argument('-w',  help='weight path',              dest='model_weight_path', type=str, default='unreal_model_weights.h5')
    parser.add_argument('-e',  help='training epochs',          dest='epochs',            type=int, default='10')
    parser.add_argument('-e2', help='fine tune epochs',         dest='epochs2',           type=int, default='2')

    args = parser.parse_args() 
    print(args)

    # Training dataset
    dataset_train = UnrealDataset()
    dataset_train.populate(args.train_object_path, args.train_images_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_valid = UnrealDataset()
    dataset_valid.populate(args.valid_object_path, args.valid_images_path)
    dataset_valid.prepare()

    # don't use all GPUs!
    limit_GPU_usage()

    # prepare the model for training
    config = UnrealConfig()
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    load_weights(model, args.init_with)

    # train the head branches: passing layers="heads" freezes all layers except the head layers.
    model.train(dataset_train, dataset_valid, 
        learning_rate=config.LEARNING_RATE, 
        epochs=args.epochs,
        layers='heads')

    # fine tuning
    model.train(dataset_train, dataset_valid, 
        learning_rate=config.LEARNING_RATE/10,
        epochs=args.epochs2,
        layers='all')

    model.keras_model.save_weights(args.model_weight_path)
    print('saved {}'.format(args.model_weight_path))


