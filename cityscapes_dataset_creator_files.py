
import json
import os
import glob
import scipy.misc
import utils
import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_dataset(annotation_folder, images_folder, list_cities=None):

    class_city = {'person':24,'rider':25,'car':26, 'truck':27, 'bus':28, 'train':31, 'motorcycle':32, 'bicycle':33, 'caravan':29, 'trailer':30,'road':7,'sidewalk':8,'parking':9,'rail track':10,'building':11,'guard rail':14,'pole':17,'traffic light':19,'traffic sign':20,'vegetation':21,'sky':23 }
    
    class_ids = {}
    for i, k in enumerate(class_city):
        class_ids[k] = i+1
    
    # main loop
    if list_cities==None:
        list_cities = glob.glob(annotation_folder+'/*')
        list_cities = list_cities[0:1]
    else:
        list_cities = [os.path.join(annotation_folder,city) for city in list_cities]
    for c in list_cities:
        city_name = c.split('/')[-1]
        print(city_name)
        json_files = glob.glob(c+'/*.json')
        for json_file in json_files:
            instances = get_instances_from_file(json_file, class_ids)
            tmp = json_file.split('/')[-1]
            cindex = tmp.find('gtFine')
            common = tmp[:cindex]
            image_name = common + 'leftImg8bit.png'
            image_path = os.path.join(images_folder, city_name, image_name )
            image = scipy.misc.imread(image_path) # open only to read the size...
            height, width, _ = image.shape
            labelIds = os.path.join(annotation_folder, city_name, common + "gtFine_labelIds.png")
            label_file = os.path.join(annotation_folder, city_name, common + "gtFine_mylabel.csv")
            instance_image_file = os.path.join(annotation_folder, city_name, common + "gtFine_mylabel.png")
            save_mask_and_label(instances,labelIds, height, width, label_file, instance_image_file, class_city)

def read_some_examples(annotation_folder, images_folder):

    # same code as in create_dataset...
    class_city = {'person':24,'rider':25,'car':26, 'truck':27, 'bus':28, 'train':31, 'motorcycle':32, 'bicycle':33, 'caravan':29, 'trailer':30,'road':7,'sidewalk':8,'parking':9,'rail track':10,'building':11,'guard rail':14,'pole':17,'traffic light':19,'traffic sign':20,'vegetation':21,'sky':23 }
    
    class_ids = {}
    class2color = {}
    colors = []
    for i, k in enumerate(class_city):
        id = i+1
        class_ids[k] = id
        color = [np.random.randint(256) for _ in range(3)]
        while color in colors:
            color = [np.random.randint(256) for _ in range(3)]
        colors.append(color)
        class2color[k] = color
    

    
    # main loop
    list_cities = glob.glob(images_folder+'/*')
    list_cities = list_cities[0:1]
    for c in list_cities:
        city_name = c.split('/')[-1]
        print(city_name)
        image_files = glob.glob(c+'/*.png')
        for image_file in image_files:
            plt.figure(1)

            # show image
            plt.subplot(131)
            image = scipy.misc.imread(image_file)
            height, width,_ = image.shape
            plt.imshow(image)

            # common name
            tmp = image_file.split('/')[-1]
            cindex = tmp.find('leftImg')
            common = tmp[:cindex]
            
            # read annotation
            label_file = os.path.join(annotation_folder, city_name, common + "gtFine_mylabel.csv")
            ids_image_file = os.path.join(annotation_folder, city_name, common + "gtFine_mylabel.png")
            annot = read_annotation(label_file) # [{'color':[R,G,B], 'label':label}]
            ids_image = cv2.imread(ids_image_file)
            
            # show instance image
            ins_image = np.zeros([height,width,3],dtype=np.uint8)
            class_image = np.zeros([height,width,3],dtype=np.uint8)
            colors = []
            for a in annot:
                # new color for object
                color = [np.random.randint(256) for _ in range(3)]
                while color in colors:
                    color = [np.random.randint(256) for _ in range(3)]
                colors.append(color)

                # obtain mask
                mask_obj = np.ones([height,width])
                for j in range(3):
                    mask_obj *= (ids_image[:,:,j] == a['color'][j])

                # fill in the channels for the instance and the classes images
                for j in range(3):
                    ins_channel = ins_image[:,:,j]
                    ins_channel[mask_obj==1] = color[j]
                    class_channel = class_image[:,:,j]
                    class_channel[mask_obj==1] = class2color[a['label']][j]  
            plt.subplot(132)
            plt.imshow(ins_image)
            plt.subplot(133)
            plt.imshow(class_image)
            plt.show()
                
        

def save_mask_and_label(instances, labelIds, height, width, label_file, instance_image_file, class_city):

    ins_image = np.zeros((height, width, 3), dtype=np.uint8)
    imgLabelIds = cv2.imread(labelIds)

    f = open(label_file, 'w')
    labels = []
    masks = np.zeros([height, width, len(instances)], dtype = np.uint8)
    for i, ins in enumerate(instances):
        id = i+1 # 0 is reserved for background
        

        #mask = np.zeros((height, width,3), dtype=np.uint8)
        #color = get_color_from_id(id)
        mask = np.zeros([height, width, 1], dtype = np.uint8)
        cv2.fillPoly(mask, np.array([ins['polygon']], np.int32), 1)
        masks[:,:,i] = mask[:,:,0]

        label = ins['label']
        labels.append(label)
        label_in_city = class_city[label]
        mask_class_in_city = imgLabelIds[:,:,0] == label_in_city
        masks[:,:,i] *= mask_class_in_city 
        #for j in range(3):
        #    mask[:,:,j] = mask[:,:,j] * mask_class_in_city

        #ins_image = ins_image + mask
        #f.write(str(int(color[0]))+','+str(int(color[1]))+','+str(int(color[2]))+','+label+'\n')

    # occlusions
    for i in range(len(instances)):
        for j in range(i):
            masks[:,:,j] *= (masks[:,:,i] ==0 )

    for i in range(len(instances)):
        id = i+1
        color = get_color_from_id(id)
        for j in range(3):
            ins_image[masks[:,:,i]==1,j] = color[j]
        f.write(str(int(color[0]))+','+str(int(color[1]))+','+str(int(color[2]))+','+labels[i]+'\n')
    
    f.close()
    cv2.imwrite(instance_image_file, ins_image)

def read_annotation(label_file):
    f = open(label_file,'r')
    ll = f.readlines()
    annot = []
    for l in ll:
        c = l.split('\n')[0].split(',')
        color = [int(c[0]), int(c[1]), int(c[2])]
        label = c[3]
        annot.append({'color':color, 'label':label})
    return annot

def get_color_from_id(id):
    color = []
    for i in range(3):
        remainder = id % 256
        color.append(remainder)
        id = id - remainder # actually this step is not needed
        id = id/256
    return color
    
def get_instances_from_file(filename, class_ids):
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
        if label in class_ids.keys():
            instances.append({'label':label, 'polygon':o['polygon']})
    f.close()
    return instances

class CityscapesDataset(utils.Dataset):
    def create_dataset(self, annotation_folder, images_folder):
        self.class_city = {'person':24,'rider':25,'car':26, 'truck':27, 'bus':28, 'train':31, 'motorcycle':32, 'bicycle':33, 'caravan':29, 'trailer':30 }
        self.my_class_ids = {}
        for i, k in enumerate(self.class_city):
            self.my_class_ids[k] = i+1
            self.add_class("cityscapes", i+1, k)

        # main loop
        list_cities = glob.glob(annotation_folder+'/*')
        #list_cities = [os.path.join(annotation_folder,city) for city in ['monchengladbach']] # TODO this has to be substituted by previous line
        for c in list_cities:
            city_name = c.split('/')[-1]
            print(city_name)
            json_files = glob.glob(c+'/*.json')
            for json_file in json_files:
                tmp = json_file.split('/')[-1]
                cindex = tmp.find('gtFine')
                common = tmp[:cindex]
                image_name = common + 'leftImg8bit.png'
                image_path = os.path.join(images_folder, city_name, image_name )
                label_path = os.path.join(annotation_folder, city_name, common + "gtFine_mylabel.csv")
                instance_image_path = os.path.join(annotation_folder, city_name, common + "gtFine_mylabel.png")
                self.add_image("cityscapes", image_id=i, path=image_path, label_path=label_path, instance_image_path=instance_image_path)

    def load_image(self, image_id):
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

        info = self.image_info[image_id]
        label_path = info['label_path']
        instance_image_path = info['instance_image_path']

        annot = read_annotation(label_path) # [{'color':[R,G,B], 'label':label}]
        ids_image = cv2.imread(instance_image_path)
        height, width, _ = ids_image.shape

        labels = []
        masks = []
        for a in annot:
            if a['label'] in self.class_city.keys():
                labels.append(a['label'])
                color = a['color']
                mask = np.zeros([height,width,1],np.uint8)
                indexes  = (ids_image[:,:,0]==color[0])
                indexes *= (ids_image[:,:,1]==color[1])
                indexes *= (ids_image[:,:,2]==color[2])
                mask[indexes,0] = 1
                masks.append(mask)

        if len(masks) == 0:
            print("empty mask on image",image_id)
            return None, None
        else:
            masks = np.concatenate(masks, 2)
            classes_ids = np.array([self.class_names.index(label) for label in labels])
            return masks, classes_ids.astype(np.int32)
        
    
    
if __name__ == "__main__":

    annotation_folder = '/home/viral/datasets/cityscapes/gtFine_trainvaltest/gtFine/train'
    images_folder = '/home/viral/datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
    
    #create_dataset(annotation_folder, images_folder)
    read_some_examples(annotation_folder, images_folder)
    
