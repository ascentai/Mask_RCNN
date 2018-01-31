import os, argparse
import model as modellib # Mask RCNN
from elecparts_config  import elecpartsConfig
from elecparts_dataset import elecpartsDataset
from unreal_utils   import MODEL_DIR, limit_GPU_usage, compute_mean_AP
from pathlib import Path
import visualize
import numpy as np

class InferenceConfig(elecpartsConfig):
    """ config for testing / prediciton / inference
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# plt
import matplotlib.pyplot as plt
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

if __name__=='__main__':
    HOME_DIR = str(Path.home())
    IMAGES_PATH      = '/nas/datashare/datasets/elecparts/samples1_test/'


    parser = argparse.ArgumentParser(description='elecparts Mask RCNN test')
    parser.add_argument('-i', help='images path', dest='test_images_path',  type=str, default=IMAGES_PATH)
    parser.add_argument('-w', help='weight path', dest='model_weight_path', type=str, default='elecparts_model_weights.h5')
    parser.add_argument('-n1', help='image count for the mAP calculation', dest='image_count_IoU',       type=int, default=300)

    parser.add_argument('-n2', help='image count to save as examples', dest='image_count_save', type=int, default=20)


    args = parser.parse_args() 

    # Test dataset
    dataset_test = elecpartsDataset()
    dataset_test.populate( args.test_images_path)
    dataset_test.prepare()

    # limit GPU usage (don't use it all!)
    limit_GPU_usage()

    # prepare the model for inference
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(args.model_weight_path, by_name=True)

    # calculate the mean average precision
    mean_AP = compute_mean_AP(model, config, dataset_test, args.image_count_IoU)
    print('Mean AP @ 0.5 IoU:', mean_AP)


    # save some examples
    for i in range(args.image_count_save):
        image_id = np.random.randint(dataset_test.num_images)
        result = modellib.load_image_gt(dataset_test, config, image_id, use_mini_mask=False)
        image, image_meta, gt_bbox, gt_mask = result
        results = model.detect([image], verbose=0)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset_test.my_class_names, r['scores'], ax=get_ax())
        result_image_result = 'results/synthetic/'+str(i).zfill(3)+".png"
        plt.savefig(result_image_result)

        
