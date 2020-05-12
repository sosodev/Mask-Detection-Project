from mrcnn.config import Config

num_train_images = 50
num_val_images = 50

class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2

    TRAIN_ROIS_PER_IMAGE = 20
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    IMAGE_RESIZE_MODE = "square"
    TRAIN_ROIS_PER_IMAGE = 20
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    STEPS_PER_EPOCH = num_train_images//(GPU_COUNT*IMAGES_PER_GPU)
    VALIDATION_STEPS = num_val_images//(GPU_COUNT*IMAGES_PER_GPU)
