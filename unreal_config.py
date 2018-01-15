from config import Config


class UnrealConfig(Config):
    """Configuration for training on the unreal dataset.
    Derives from the base Config class and overrides values specific
    to this dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Unreal"

    CLASS_NAMES = ['Pedestrian', 'Vehicles']
    NUM_CLASSES = 1 + len(CLASS_NAMES) # background + normal classses
    CLASS_NAME_TO_ID = {class_name:str(i+1) for i, class_name in enumerate(CLASS_NAMES)}

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + NUM_CLASSES  # background + normal classses

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STPES = 5

