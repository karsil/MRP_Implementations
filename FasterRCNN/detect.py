#! /usr/bin/env python
import os
import numpy as np
import pathlib
import tarfile
from PIL import Image
import tensorflow as tf

import visualization_utils as vis_util
import label_map_util

PATH_TO_LABELS = './mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

print("TensorFlow version: ", tf.__version__)

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'

    downloadFolder = os.getcwd() + '/downloaded_models/'

    if not os.path.exists(downloadFolder):
        os.makedirs(downloadFolder)

    print("Checking if not exists: " + downloadFolder + model_name)
    if not os.path.exists(downloadFolder + model_name):
        # TODO. Also check if archive has been downloaded, but not extracted
        model_dir = tf.keras.utils.get_file(
            fname=downloadFolder + model_file,
            origin=base_url + model_file
            )
        print("Extracting at " + downloadFolder + model_name)
        tar = tarfile.open(model_dir, "r:gz")
        tar.extractall(downloadFolder)
        tar.close()
        print("Extracted")

    else:
        model_dir = pathlib.Path(downloadFolder + model_name)

    model_dir = downloadFolder + model_name
    model_dir = pathlib.Path(model_dir)/"saved_model"
    model_dir = pathlib.Path(model_dir)
    print("model_dir:")
    print(model_dir)

    model = tf.saved_model.load(export_dir=str(model_dir), tags=None)
    model = model.signatures['serving_default']

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))

  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  img = Image.fromarray(image_np)
  img.save('result.png')

if __name__ == '__main__':
    # patch tf1 into `utils.ops`
    #utils_ops.tf = tf.compat.v1

    # Patch the location of gfile (needed in label_map_util, does not overwrite right know?)
    #tf.gfile = tf.io.gfile

    #model_name = 'faster_rcnn_resnet101_coco_11_06_2017'
    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    detection_model = load_model(model_name)

    print("inputs: ", detection_model.inputs)

    print("detection_model.output_dtypes: ", detection_model.output_dtypes)

    print("detection_model.output_shapes: ", detection_model.output_shapes)

    image_path = './images/dogs.jpg'
    show_inference(detection_model, image_path)

