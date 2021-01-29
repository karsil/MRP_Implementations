checkpoint_dir=/home/jsteeg/Tensorflow/tf2-object-detection-api-tutorial/out/
out_dir=/home/jsteeg/Tensorflow/tf2-object-detection-api-tutorial/out/eval-train
mkdir -p $out_dir

config=/home/jsteeg/Tensorflow/tf2-object-detection-api-tutorial/models/my_frcnn_50_1024/pipeline.config

# uncomment this to run the test on CPU
#export CUDA_VISIBLE_DEVICES="-1"
export CUDA_VISIBLE_DEVICES="0"

python model_main_tf2.py --alsologtostderr --model_dir=$out_dir \
                         --pipeline_config_path=$config \
                         --eval_on_train_data=True \
                         --sample_1_of_n_eval_on_train_examples=20 \
                         --checkpoint_dir=$checkpoint_dir  2>&1 | tee $out_dir/eval_traindata.log
