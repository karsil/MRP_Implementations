# Implementions for Faster-RCNN
## Installation of dependencies
First, you need to clone the **Tensorflow models** repository:
```bash
git clone https://github.com/tensorflow/models
```
Then add the path to tensorflow models for accessing the libraries:
```bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
Then execute the Protobuf Compilation
```bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```
(Further information about installing TensorFlow: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

Back in your local clone of this repository, you can now install the local dependencies:
```bash
pip3 install -r requirements.txt
```

## Run inference
Check ``core/config.py`` and modify paths and model names. Then:

```bash
python3 detect.py
```
As a result, a pretrained Faster-RCNN model is downloaded into the folder ``pretrained_models`` and executed on sample images.
Content of this folder is used for later training.

## Train
If you use greyscale images (one channel), you need to transform those into RGB images (three channels).
You can use the following script to achieve this:
```bash
python utils/conversion_util.py -in greyscale_images -out rgb_images
```

Check ``core/config.py`` and modify paths to your own system (and possible transformed RGB images) and desired model names, then you can convert your training data to TFRecords:
```bash
python create_tf_record.py
```
Resulting TFRecords are stored in folder ``data``.
Copy your ``annotations.txt`` (it should contain absolute paths to the RGB images), ``classes`` and (based on ``classes``) your ``label_map.pbtxt`` into ``data``

Directory ``training`` needs to contain ``pipeline.config``, ``frozen_inference_graph.pb`` (both in the download folder ``pretrained_models``) and two empty folders ``eval`` and ``train``.
Pipeline
``pipeline.config`` can be downloaded in its latest version [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) and has to be modified:
1. Change ``num_classes`` to the amount of classes of your dataset.
2. Replace the following part containing ``step: 0`, if existing:
```
schedule {
    step: 0
    learning_rate: ...
}

```
3. Replace all ``PATH_TO_BE_CONFIGURED`` with the absolute paths of the files mentioned above
    - .ckpt in ``pretrained_models``
    - record files to the corresponding generated tf_record files in ``data``
    - label map in ``data``

Modify variables in ``train.sh``, then
```bash
./train.sh
```

After training, we can export the final checkpoint to the

```bash
python3 /PATH/TO/TF/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```