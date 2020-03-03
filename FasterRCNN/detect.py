#! /usr/bin/env python
# Detector using the TensorFlow Object Detection API

import os
import numpy as np
import pathlib
import tarfile
from PIL import Image
import tensorflow as tf


from core.config import cfg

from utils import label_map_util, visualization_utils as vis_util
from utils import dataset_util

print("Using TensorFlow version: ", tf.__version__)

# Setting up labels of trained model
PATH_TO_LABELS = cfg.PRETRAINED_MODEL_LABELS
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

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

def download_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'

    downloadFolder = os.getcwd() + cfg.DOWNLOADED_MODELS_FOLDER

    if not os.path.exists(downloadFolder):
        os.makedirs(downloadFolder)

    if not os.path.exists(downloadFolder + model_name):
        print(model_name + " does not exists yet, downloading...")
        # TODO. Also check if archive has been downloaded, but not extracted yet
        model_dir = tf.keras.utils.get_file(
            fname=downloadFolder + model_file,
            origin=base_url + model_file
            )
        print("Download done, extracting at " + downloadFolder + model_name + "...")
        tar = tarfile.open(model_dir, "r:gz")
        tar.extractall(downloadFolder)
        tar.close()
        print("Extracted, removing archive...")
        os.remove(downloadFolder + model_file)
        print("Archive successfully removed.")

    else:
        model_dir = pathlib.Path(downloadFolder + model_name)

    # Subfolder of model contains SavedModel
    model_dir = downloadFolder + model_name
    #model_dir = pathlib.Path(model_dir)/cfg.SAVED_MODEL_SUBFOLDER
    #model_dir = pathlib.Path(model_dir)
    print("Using model in " + downloadFolder + model_name)

    return model_dir

def load_model(model_dir):
    model_dir = pathlib.Path(model_dir) / cfg.SAVED_MODEL_SUBFOLDER
    print("Loading saved model at ", model_dir)
    model = tf.compat.v2.saved_model.load(export_dir=str(model_dir), tags=None)
    model = model.signatures['serving_default']
    return model

def import_graph(model_dir):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        PATH_TO_CKPT = model_dir + "/frozen_inference_graph.pb"
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph

def run_inference_for_single_image(model, image):
    image = np.array(Image.open(image))
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


def draw_bboxes(image_path, output_dict):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image = np.array(Image.open(image_path))

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

    return image


def run_and_draw_bboxes(model, image_path):
    output_dict = run_inference_for_single_image(model, image_path)
    return draw_bboxes(image_path, output_dict)

def save_image(image, name):
    img = Image.fromarray(image)
    outputFolder = os.getcwd() + cfg.RESULT_FOLDER
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    outputPath = outputFolder + name
    img.save(outputPath)
    print("Result stored at " + outputPath)

# for tf2
def runSavedModel(detection_model):
    # patch tf1 into `utils.ops`
    #utils_ops.tf = tf.compat.v1

    # Patch the location of gfile (needed in label_map_util, does not overwrite right know?)
    #tf.gfile = tf.io.gfile

    # print("inputs: ", detection_model.inputs)
    # print("detection_model.output_dtypes: ", detection_model.output_dtypes)
    # print("detection_model.output_shapes: ", detection_model.output_shapes)

    PATH_TO_TEST_IMAGES_DIR = pathlib.Path(cfg.IMAGES_PATH)
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*." + cfg.IMAGES_TYPE)))

    for image_path in TEST_IMAGE_PATHS:
        _, image_name = os.path.split(image_path)
        output_dict = run_inference_for_single_image(detection_model, image_path)
        processedImage = draw_bboxes(image_path, output_dict)
        save_image(processedImage, image_name)

def run_by_checkpoint_v1(detection_graph):
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path(cfg.IMAGES_PATH)
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*." + cfg.IMAGES_TYPE)))

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                _, image_name = os.path.split(image_path)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                save_image(image_np, image_name)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  if image.mode is 'RGB':
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  elif image.mode is 'L':
      # copy greyscale layer two times and stack all those to result in RGB
      image = np.stack((image.getdata(),)*3, axis=-1)
      return image.reshape((im_height, im_width, 3)).astype(np.uint8)

if __name__ == '__main__':

    model_path = download_model(cfg.PRETRAINED_MODEL_NAME)

    graph = import_graph(model_path)

    run_by_checkpoint_v1(graph)


