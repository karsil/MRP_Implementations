#!/bin/bash
PIPELINE_CONFIG_PATH=./training/pipeline.config
MODEL_DIR=./training
NUM_TRAIN_STEPS=10
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 ../../MASTERPROJECT/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
