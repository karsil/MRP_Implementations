# Implementions for Faster-RCNN
## Installation of dependencies
```
pip3 install -r requirements.txt
```

Additional, you need to clone the **Tensorflow models** repository:
```
git clone https://github.com/tensorflow/models
```
Then add the path to tensorflow models for accessing the libraries:
```
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
Further information about installing TensorFlow: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

## Run inference
Check ``core/config.py`` and modify paths and model names. Then:

```python
python3 detect.py
```

## Train
Modify variables in ``train.sh``, then:

```bash
./train.sh
```