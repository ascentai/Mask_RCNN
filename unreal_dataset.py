"""
Dataset class used by Mask R-CNN model for UNREAL synthetic images
"""
import os, glob, json, imageio
import numpy as np
import utils # Mask RCNN
from unreal_config import UnrealConfig


class UnrealDataset(utils.Dataset):
    """ This represents the images (the original and the instance segmentation) used by Mask R-CNN
    """

    def populate(self, images_path, object_descriptions_path):
        """ This method must be called after the initializaiton and before the prepare method
        """
        self.add_classes()
        self.load_object_descriptions(object_descriptions_path)
        self.load_image_annotations(images_path, images_path)
    
    def add_classes(self):
        """ add classes we care
        """
        for class_name, class_id in UnrealConfig.CLASS_NAME_TO_ID.items():
            self.add_class(UnrealConfig.NAME, class_id, class_name)

    def load_object_descriptions(self, path):
        """ load object descriptions to get the color and the class of instance segementation
        """
        self.descriptions = {}
        with open(path) as f:
            for color, attributes in json.loads(f.read()).items():
                instance_id = int(attributes['id'])
                class_name = attributes['class']
                description = {}
                description['id']         = instance_id
                description['name']       = attributes['name']
                description['class_id']   = UnrealConfig.CLASS_NAME_TO_ID.get(class_name, 0) # 0 for BG
                description['class_name'] = class_name
                description['color']      = color
                description['rgb']        = [int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)]
                self.descriptions[instance_id] = description

    def load_image_annotations(self, image_folder, annotation_folder):
        """ load instance segmentation images
        """
        for i, image_path in enumerate(glob.glob(image_folder+'/lit*.png')):
            # obtain the number of the image. image_path = 'folder/lit_000001.png'
            ending = image_path.split('_')[-1] # '000001.png'
            instance_path = annotation_folder + '/mask_' + ending
            self.add_image(UnrealConfig.NAME,
                           image_id=i,
                           path=image_path,
                           instance_path=instance_path)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        image = imageio.imread(image_path)
        image = image[:, :, :3] # discard alpha
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == UnrealConfig.NAME:
            return info["path"]
        return super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]

        # instance image
        instance_path = info['instance_path']
        instance_image = imageio.imread(instance_path)
        height, width, _ = instance_image.shape

        # brute-force.. TODO improve
        masks       = []
        class_ids   = []
        for _, description in self.descriptions.items():
            rgb = description['rgb']
            mask = np.zeros([height,width,1], np.uint8)
            indexes  = (instance_image[:,:,0]==rgb[0])
            indexes *= (instance_image[:,:,1]==rgb[1])
            indexes *= (instance_image[:,:,2]==rgb[2])
            if np.sum(indexes) > 0:
                mask[indexes,0] = 1
                masks.append(mask)
                if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 32: # TODO skipping the road which has low blue only
                    class_ids.append(0)
                else:
                    class_ids.append(description['class_id'])
        
        if len(masks) == 0:
            print("empty mask on image",image_id)
            return None, None
        
        masks = np.concatenate(masks, 2)                
        return masks, np.array(class_ids, np.int)
    
