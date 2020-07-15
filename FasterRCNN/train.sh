#!/bin/bash
PIPELINE_CONFIG_PATH=./model/pipeline.config
MODEL_DIR=./model
NUM_TRAIN_STEPS=20
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 ../../models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
