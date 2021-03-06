#! /usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict

__C                             = edict()
cfg                             = __C

# Faster RCNN options
__C.FASTERRCNN                 = edict()
__C.PRETRAINED_MODEL_NAME      = 'faster_rcnn_resnet50_coco_2018_01_28'
__C.PRETRAINED_MODEL_LABELS    = './mscoco_label_map.pbtxt'

__C.TRAINED_MODEL_PATH         = "/home/jsteeg/MRP_Implementations/FasterRCNN/inference_graph"


__C.DOWNLOADED_MODELS_FOLDER   = '/pretrained_models/'
__C.SAVED_MODEL_SUBFOLDER      = 'saved_model'
__C.RESULT_FOLDER              = "/results/"

__C.IMAGES_PATH                = './images'
__C.IMAGES_TYPE                = 'jpg'

# Set the class name
__C.FASTERRCNN.CLASSES          = "/home/jsteeg/ufo_data/yolo_no_crop_vc/classes"
__C.FASTERRCNN.LABELMAP         = "/home/jsteeg/ufo_data/yolo_no_crop_vc/label_map.txt"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "/home/jsteeg/ufo_data/yolo_no_crop_vc/train.txt"
__C.TRAIN.RECORDS               = "/home/jsteeg/ufo_data/yolo_no_crop_vc/train.tfrecord"

# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH            = "/home/jsteeg/ufo_data/yolo_no_crop_vc/test.txt"
__C.TEST.RECORDS               = "/home/jsteeg/ufo_data/yolo_no_crop_vc/test.tfrecord"

