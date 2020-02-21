#! /usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict

__C                             = edict()
cfg                             = __C

# Faster RCNN options
__C.FASTERRCNN                 = edict()
__C.PRETRAINED_MODEL_NAME      = 'faster_rcnn_inception_v2_coco_2018_01_28'
__C.PRETRAINED_MODEL_LABELS    = './mscoco_label_map.pbtxt'

__C.DOWNLOADED_MODELS_FOLDER   = '/pretrained_models/'
__C.SAVED_MODEL_SUBFOLDER      = 'saved_model'
__C.RESULT_FOLDER              = "/results/"

__C.IMAGES_PATH                = './images'
__C.IMAGES_TYPE                = 'jpg'

# Set the class name
__C.FASTERRCNN.CLASSES          = "/home/jsteeg/ufo_data/set01/class_names"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "/home/jsteeg/ufo_data/yolo/datasetAbsPath.txt"

# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "/home/jsteeg/ufo_data/yolo/datasetAbsPath.txt"






