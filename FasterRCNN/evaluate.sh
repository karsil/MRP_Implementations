#!/bin/bash
PIPELINE_CONFIG_PATH=./model/faster_rcnn_inception_v2_ufo.config
CHECKPOINT_DIR=inference_graph/
MODEL_DIR=eval/

# legacy
python3 ~/models/research/object_detection/legacy/eval.py \
    --logtostderr \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --eval_dir=${EVAL_DIR} \
    --eval_config_path=${EVAL_CONFIG} \
