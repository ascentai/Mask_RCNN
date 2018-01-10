import json # the file that will contain the correspondences between ids, color and class will be provided in json format. Michele was working on this.
import utils
import glob
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

class UnrealDataset(utils.Dataset):
    
    def create_dataset(self, annotation_folder, images_folder):
        '''
        images_folder: where the "lit" images (real ones) can be found. The images have the same name format 'lit_123456.png' (6 digits for the image number)
        annotation_folder: where the annotation files are found. The annotation files associated with 'lit_123456.png' are 'objectmask_123456.png' and 'annotation_123456.csv'
        'objectmask_123456.png' is an image in which each all the pixels of the different objects have the color to be found in all_objects.csv.
        'annotation_123456.csv' contains in the first line the pose of the camera (x,y,z,Rx,Ry,Rz). Each line from the second line onwards contains the pose of an object. Each of these lines contains: object_id, x,y,z, Rx,Ry, Rz. The 
        The set of objects in this file should contain at least all the objects of the type of interest that appear in the image. For example, this list objects could correspond to all objects of the world, not only the rendered ones.
        The annotation_folder also contains 'all_objects.csv'
        all_objects.csv: each line corresponds to one object in the world.
        The format of each line is: id,R,G,B,class,lx,ly,lz. Where id is the unique id of the object, R,G,B is the decimal representation of the color of the object's pixels to be seen in the objectmask_xxxxxx.png images, class is the string that 
        describes the class of the object ('person' or 'vehicle')
        '''
        # classes
        self.class_unreal = {'person':1, 'vehicle':2}
        self.my_class_ids = {}
        for i, k in enumerate(self.class_unreal):
            self.my_class_ids[k] = i+1
            self.add_class("unreal", i+1, k)

        # TODO: read the file containing the instances' (vehicles and persons) dimensions, color, class and ID.
        self.all_objects = self.read_all_objects(annotation_folder + '/all_objects.csv') # { id:{color, class, dimensions},...}

        # main loop
        list_images = glob.glob(images_folder+'/*.png')
        for i, image_path in enumerate(list_images):
            # obtain the number of the image. image_path = 'folder/lit_000001.png'
            ending = image_path.split('_')[-1] # '000001.png'
            instance_path = annotation_folder + '/objectmask_' + ending
            annotation_path = annotation_folder + '/annotation_' + ending.split('.')[0] + '.csv'
            camera_pose, annotation_list = self.read_annotation(annotation_path) # [x,y,z,Rx,Ry,Rz], [[id, x,y,z,Rx,Ry,Rz], ...]
            # TODO: OPTIONALLY, if not too much space is needed, save also the masks
            self.add_image("unreal", image_id=i, path = image_path, instance_path = instance_path, annotation_path = annotation_path, annotation_list = annotation_list, camera_pose = camera_pose)

            
    def load_image(self, image_id):
        '''
        load an image by id
        '''
        info = self.image_info[image_id]
        image_path = info['path']
        return scipy.misc.imread(image_path)

    
    def image_reference(self, image_id):
        """Returns the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "unreal":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

            
    def load_mask(self, image_id):
        '''
        load a mask by id
        '''
        info = self.image_info[image_id]

        # instance image
        instance_path = info['instance_path']
        instance_image = scipy.misc.imread(instance_path)
        height, width, _ = instance_image.shape

        # annotation
        annot_list = info['annotation_list'] #  [[id, x,y,z,Rx,Ry,Rz], ...]

        labels = []
        masks = []
        for a in annot_list:
            id = a[0]
            label = self.all_objects[id]['class']
            if label in self.class_unreal.keys():
                labels.append(self.class_unreal[label])
                color = self.all_objects[id]['color']
                mask = np.zeros([height,width,1],np.uint8)
                indexes  = (instance_image[:,:,0]==color[0])
                indexes *= (instance_image[:,:,1]==color[1])
                indexes *= (instance_image[:,:,2]==color[2])
                mask[indexes,0] = 1
                masks.append(mask)

        if len(masks) == 0:
            print("empty mask on image",image_id)
            return None, None
        else:
            masks = np.concatenate(masks, 2)
            classes_ids = np.array(labels)
            return masks, classes_ids.astype(np.int32)

    def read_all_objects(self, path):
        '''
        read all_object.csv
        '''
        f = open(path)
        ll = f.readlines()
        all_objects = {}
        for l in ll:
            a = []
            b = l.split('\n')[0].split(',')
            id = int(b[0])
            a = {'color':[int(b[1]), int(b[2]), int(b[3])], 'class': b[4], 'dimensions':[float(b[5]), float(b[6]), float(b[7])]}
            all_objects[id] = a
        f.close()
        return all_objects

    def read_annotation(self, path):
        '''
        read annotation_123456.csv
        '''
        with open(path) as f:
            ll = f.readlines()
            # camera pose
            camera_pose = ll[0].split('\n')[0].split(',')
            camera_pose = [ float(x) for x in camera_pose]

            # each object pose
            all_objects_pose = []
            for i in range(1, len(ll)):
                object_pose = ll[i]
                object_pose = object_pose.split('\n')[0].split(',')
                object_pose = [int(object_pose[0])] + [float(x) for x in object_pose[1:] ] # [id, x,y,z, Rx, Ry, Rz]
                all_objects_pose.append(object_pose)

        return camera_pose, all_objects_pose
            
            
        
    def show_some_example(self, id=0):
        '''
        plot in a 2x2 grid of images, the lit image (TopLeft), the whole instance image (TR), the first object mask (BL), the second object mask (BR)
        '''

        image = self.load_image(id)
        masks, classes_ids = self.load_mask(id)
        print(masks.shape)
        
        info = self.image_info[id]
        #image_path = info['path']
        #image = scipy.misc.imread(image_path)
        instance_path = info['instance_path']
        instance_image = scipy.misc.imread(instance_path)

        print(info['annotation_list'])
        for a in info['annotation_list']:
            print(a[0], self.all_objects[a[0]]['color'])

        plt.subplot(221)
        plt.imshow(image)
        plt.subplot(222)
        plt.imshow(instance_image)
        plt.subplot(223)
        plt.imshow(masks[:,:,0])
        plt.subplot(224)
        plt.imshow(masks[:,:,1])

        plt.show()

if __name__ == "__main__":

    annotation_folder = '/home/fernando/datasets/unreal/train/annot'
    images_folder =     '/home/fernando/datasets/unreal/train/images'

    dataset = UnrealDataset()
    dataset.create_dataset(annotation_folder, images_folder)
    dataset.show_some_example(0)
