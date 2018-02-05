import imageio
import argparse

PARSER = argparse.ArgumentParser("test a model in a real set of images")
PARSER.add_argument("--images_folder", type=str, required=True, help="where to find the test images (must include the final '/' in the path)")
PARSER.add_argument("--model", type=str, required=True, help="model that contains the trained weights")
PARSER.add_argument("--output_folder",type=str, default="./results/",help="where the result images will be placed")

args = PARSER.parse_args()

# images
import glob
images_names =  glob.glob(args.images_folder+'*.jpg')
images_names += glob.glob(args.images_folder+'*.png')

# config
from elecparts_config import elecpartsConfig
class InferenceConfig(elecpartsConfig):
    """ config for testing / prediciton / inference
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()

# model and weights
import model as modellib
from unreal_utils   import MODEL_DIR, limit_GPU_usage, compute_mean_AP
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(args.model, by_name=True)



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
# visualize
import visualize

# dataset
from elecparts_dataset import elecpartsDataset
dataset = elecpartsDataset()



# main loop
import numpy as np
from skimage import transform as tf
for path in images_names:

    print(path)
    
    # rotate and resize if necessary
    img = imageio.imread(path)
    height, width, _ = img.shape
    if height > width:
       img = np.swapaxes(img, 0, 1)
       img = tf.resize(img, [768,1024])*255
       img = img.astype('uint8')
    
    
    results = model.detect([img], verbose=0)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], dataset.my_class_names, r['scores'], ax=get_ax())
    image_name = path.split('/')[-1]
    result_image_result = args.output_folder+image_name
    plt.savefig(result_image_result)
    plt.show()
