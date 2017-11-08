
import json
import os
import glob
import scipy.misc
import utils
import numpy as np
import cv2
import matplotlib.pyplot as plt


class CityscapesDataset(utils.Dataset):
    """Generates the Cityscapes dataset.
    The dataset access to the images stored in the disk when load_image is called.
    The masks are generated on the fly. Then reason to do this instead of accessing a set of files for the masks is that the masks are individiually defined by each instance in the image with the same full size image. Since there are can be many instances in an image it is more space-efficient to genereate these masks on the fly. In connection to this and in order to save memory in the GPU while training,we can generate a configuration object for the MaskRCNN that allows to use mini masks (while training it will crop the masks to their instance bboxes and resize the mask to a certain size defined in the same configuration object).
    In the construction we have to give two paths, the path where to find the folders containing the annotation for each city and the pathe path of the folder that contains the corresponding images separated by city folders.
    
    """

    def create_dataset(self, annotation_folder, images_folder):
        """
        images_folder should be an absolute path
        """
        
        assert images_folder[0] == '/'
        assert annotation_folder[0] == '/'
        
        # Add classes
        #self.class_city = {'person':24,'rider':25,'car':26, 'truck':27, 'bus':28, 'train':31, 'motorcycle':32, 'bicycle':33, 'caravan':29, 'trailer':30, 'road':7, 'sidewalk':8, 'parking':9,'pole':17, 'traffic light':19, 'traffic sign':20 }
        self.class_city = {'person':24,'rider':25,'car':26, 'truck':27, 'bus':28, 'train':31, 'motorcycle':32, 'bicycle':33, 'caravan':29, 'trailer':30 }
        # TODO add class for other instances not so "instance-type" like sky, road, side walk...
        self.class_ids = {}
        for i, k in enumerate(self.class_city):
            self.class_ids[k] = i+1
            self.add_class("cityscapes", i+1, k)

        # main loop
        list_cities = glob.glob(annotation_folder+'/*')
        list_cities = list_cities[0:1] # TODO: REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!
        for c in list_cities:
            city_name = c.split('/')[-1]
            print(city_name)
            json_files = glob.glob(c+'/*.json')
            for json_file in json_files:
                instances = self.get_instances_from_file(json_file)
                tmp = json_file.split('/')[-1]
                cindex = tmp.find('gtFine')
                common = tmp[:cindex]
                image_name = common + 'leftImg8bit.png'
                image_path = os.path.join(images_folder, city_name, image_name )
                image = scipy.misc.imread(image_path)
                height, width, _ = image.shape
                labelIds = os.path.join(annotation_folder, city_name, common + "gtFine_labelIds.png")
                self.add_image("cityscapes", image_id=i, path=image_path, instances=instances, width=width, height=height, labelIds=labelIds)
                
                
                
    def get_instances_from_file(self, filename):
        """
        returns a list of elements of the form {'label':class_label, 'polygon':[[x1,y1], [x2,y2],...]}
        filename is a the name of a json file as the one used in cityscapes.
        """
        f = open(filename)
        data = json.load(f)
        instances = []
        objects = data['objects']
        for o in objects:
            label = o['label']
            if label.endswith("group"):
                label = label[:-len("group")]
            if label in self.class_ids.keys():
                instances.append({'label':label, 'polygon':o['polygon']})
        f.close()
        return instances

    def load_image(self, image_id):
        """
        load the image from the disk
        """
        info = self.image_info[image_id]
        image_path = info['path']
        return scipy.misc.imread(image_path)
        
    def image_reference(self, image_id):
        """Returns the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cityscapes":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        labelIds = info['labelIds']
        #print(labelIds)
        imgLabelIds = cv2.imread(labelIds)
        instances = info['instances']
        count = len(instances)
        masks = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, ins in enumerate(instances):
            label = ins['label']
            cityId = self.class_city[label]
            mask = np.zeros((info['height'], info['width']), dtype=np.uint8)
            cv2.fillPoly(mask, np.array([ins['polygon']],np.int32), 1) # fillpoly expects an array of polygon which are also an array of points.
            mask = mask * (imgLabelIds[:,:,0] == cityId) # We remove the pixels that have a different class id.
            masks[:,:,i] = mask

        # remove occlusions. We assume that any object in the list of instances with ith index is closer plane to the camera for any other instance with index j such that j<i.
        for i in range(1,len(instances)):
            for j in range(i):
                masks[:,:,j] *= (masks[:,:,i] == 0 )
       
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(ins['label']) for ins in instances])
        #print("dataset: masks:",str(masks.shape),str(class_ids))
        return masks, class_ids.astype(np.int32)

    def show_masks(self, image_id):
        masks, class_ids = self.load_mask(image_id)
        height, width, num_instances = masks.shape
        img = np.zeros([height, width,3])
        
        for i in range(num_instances):
            mask = masks[:,:,i]
            color = [np.random.randint(256) for j in range(3)]
            mask_exp = np.expand_dims(mask,2)
            mask_img = np.tile(mask_exp,3)
            for j in range(3):
                mask_img[:,:,j] *= color[j]
            img += mask_img
        plt.imshow(img)


if __name__ == "__main__":
    dataset = CityscapesDataset()
    dataset.create_dataset('/home/viral/datasets/cityscapes/gtFine_trainvaltest/gtFine/train', '/home/viral/datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train')
    dataset.prepare()
    img = dataset.load_image(0)
    for i in range(0):
        img = dataset.load_image(i)
        fig = plt.figure()
        a = fig.add_subplot(2,1,1)
        plt.imshow(img)
        a = fig.add_subplot(2,1,2)
        dataset.show_masks(i)
        plt.show()
