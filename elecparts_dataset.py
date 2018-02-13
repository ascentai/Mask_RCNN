import os, glob, json, imageio
import numpy as np
import utils # Mask RCNN
from elecparts_config import elecpartsConfig


class elecpartsDataset(utils.Dataset):
    """ This represents the images (the original and the instance segmentation) used by Mask R-CNN
    """
    def __init__(self):
        super(elecpartsDataset, self).__init__()
        self.add_classes()

    def populate(self, images_path):
        """ This method must be called after the initializaiton and before the prepare method
        """
        self.load_images_info(images_path)
    
    def add_classes(self):
        """ add classes we care
        """
        #self.my_class_names = ['BG', 'none', 'none', 'none', 'none']
        self.my_class_names = ['BG', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']
        for class_name, class_id in elecpartsConfig.CLASS_NAME_TO_ID.items():
            self.add_class(elecpartsConfig.NAME, class_id, class_name)
            self.my_class_names[int(class_id)] = class_name


    def load_images_info(self, images_folder):
        """ load instance segmentation images
        """
        for i, image_path in enumerate(glob.glob(images_folder+'/scene*.png')):
            # obtain the number of the image. image_path = 'folder/scene_0000.png'
            ending = image_path.split('_')[-1] # '0000.png'
            scene_id = int(ending.split('.png')[0])
            self.add_image(elecpartsConfig.NAME,
                           image_id=i,
                           path=image_path,
                           scene_id=scene_id)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        image = imageio.imread(image_path)
        image = image[:, :, :3] # discard alpha if any
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == elecpartsConfig.NAME:
            return info["path"]
        return super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        scene_path = info['path']
        scene_id = info['scene_id']
        common_objs_path = scene_path.split('scene')[0]+"object_"+str(scene_id).zfill(4)+"_"        

        masks = []
        class_ids = []
        for i in range(elecpartsConfig.NUM_CLASSES-1): # It would be better to have a file telling us which objects are present in the scene, specially when we have many possible objects.
            path = common_objs_path +str(i).zfill(2)+".png"
            mask_image = imageio.imread(path)
            height, width, _  = mask_image.shape
            mask = np.zeros([height,width,1], np.uint8)
            indexes = mask_image[:,:,0] > 200
            if np.sum(indexes) > 0:
                mask[indexes,0] = 1
                masks.append(mask)
                class_ids.append(i+1)
        
        if len(masks) == 0:
            print("empty mask on image",image_id)
            return None, None
        else:
            masks = np.concatenate(masks, 2)                
            return masks, np.array(class_ids, np.int)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import visualize
    
    dataset = elecpartsDataset()
    dataset.populate("/nas/datashare/datasets/elecparts/samples1")
    dataset.prepare()

    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.my_class_names)
    plt.show()
