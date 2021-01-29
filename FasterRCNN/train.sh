#!/bin/bash
PIPELINE_CONFIG_PATH=./tf2/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8/pipeline.config
MODEL_DIR=./tf2/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu/train/
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
NUM_TRAIN_STEPS=60000
NUM_TRAIN_STEPS=20
python3 ~/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
