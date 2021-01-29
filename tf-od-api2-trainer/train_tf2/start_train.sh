out_dir=/home/jsteeg/Tensorflow/tf2-object-detection-api-tutorial/out
mkdir -p $out_dir

#export CUDA_VISIBLE_DEVICES="0"
export CUDA_VISIBLE_DEVICES="1"

# divide size of train data by batch size in config (8980 / 2)
steps_per_epoch=4490

config=/home/jsteeg/Tensorflow/tf2-object-detection-api-tutorial/models/my_frcnn_50_1024/pipeline.config

python model_main_tf2.py \
	--alsologtostderr \
	--model_dir=$out_dir \
	--checkpoint_every_n=$steps_per_epoch \
	--pipeline_config_path=$config \
#	--sample_1_of_n_eval_examples=1 \
#    | tee $out_dir/train.log


# --eval_on_train_data 2>&1 

# to try:
# --num_train_steps=50000