#!/bin/bash
PIPELINE_CONFIG_PATH=./model/faster_rcnn_inception_v2_ufo.config
CHECKPOINT_DIR=inference_graph/
MODEL_DIR=eval/

python3 ~/models/research/object_detection/model_main.py \
    --logtostderr \
    --model_dir=${MODEL_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --run_once=true