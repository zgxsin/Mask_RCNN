"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os import listdir
from os.path import isfile, join
import cv2
# import os
import tarfile
import shutil
from PIL import Image


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CilConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cil_project"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    GPU_COUNT = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + road

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    VALIDATION_STEPS = 1

    # set MINI mask shape, since tram is always rectangular.
    # MINI_MASK_SHAPE = (56, 90)


############################################################
#  Dataset
############################################################

class CilDataset(utils.Dataset):

    def load_cil(self, original_directory, subset):
        '''

        :param dataset_dir: working directory: /scratch/zgxsin/dataset/; unzip orignal data to "unzip_directory" where we will be working
        :param subset: train/val
        :param unzip_directory:
        :return:
        '''
        """Load a subset of the cil_project dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cil_project", 1, "road")

        # Train or validation dataset?
        # In our case, there is not valiadation set.
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(original_directory, subset)
        mask_path = os.path.join(dataset_dir, "groundtruth")

        ##############
        ############

        #############
        mask_list = [f for f in listdir(mask_path) if isfile(join(mask_path,f))]
        image_counter = 0

        for i, filename in enumerate(mask_list):
            image_path = os.path.join(dataset_dir,"images",filename)
            try:
                image = skimage.io.imread(image_path)
            except:
                print("Not a valid path: ", image_path)
                continue
            height, width = image.shape[:2]
            mask_temp = skimage.io.imread(os.path.join(mask_path, filename), as_grey=True)
            # mask has to be bool type

            mask_temp = mask_temp > 0
            masks = np.asarray(mask_temp, np.bool)
            masks = np.expand_dims(masks, 0)
            self.add_image(
                "cil_project",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=masks)
            image_counter = image_counter+1

        string = "trainging" if subset=="train" else "validation"
        print("The number of {0} samples is {1} at Cil Dataset".format(string, image_counter))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cil_project":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = info["polygons"].reshape(info["height"], info["width"], -1)
        mask_accu = info["polygons"]
        mask = np.empty(shape=(info["height"], info["width"], mask_accu.shape[0]), dtype=np.bool)
        for i in range(mask_accu.shape[0]):
            mask[:,:,i] = mask_accu[i,:,:]

        if mask is not None:
            return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        else:
            return mask.astype(np.bool), np.empty([0], np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cil_project":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    # to handle broken pipe error
    from signal import signal, SIGPIPE, SIG_DFL
    signal( SIGPIPE, SIG_DFL )
    #####################
    # handle directories
    #####################
    # codes for handling cil images because of Leonhard limitation
    #####################
    # handle directories
    #####################


    dataset_train = CilDataset()
    dataset_train.load_cil(args.dataset, "train")
    dataset_train.prepare()


    # Validation dataset
    dataset_val = CilDataset()
    dataset_val.load_cil(args.dataset, "val")
    dataset_val.prepare()


    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")


    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=70,
                layers='heads')

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cil/dataset/",
                        help='Directory of the CIL dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments

    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CilConfig()
    else:
        class InferenceConfig(CilConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
