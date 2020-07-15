#!/bin/bash
PIPELINE_CONFIG_PATH=./model/faster_rcnn_inception_v2_ufo.config
MODEL_DIR=./model
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
NUM_TRAIN_STEPS=20
python3 ~/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
