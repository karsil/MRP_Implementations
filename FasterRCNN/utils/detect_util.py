from core.config import cfg
import numpy as np
import tensorflow as tf
import pathlib
import os

import tarfile
from datetime import datetime
from PIL import Image

from utils import conversion_util
from utils import label_map_util, visualization_utils as vis_util

# Setting up labels of trained model
PATH_TO_LABELS = cfg.PRETRAINED_MODEL_LABELS
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
from core.config import cfg
import numpy as np
import tensorflow as tf
import pathlib
import os

import tarfile
from PIL import Image

from utils import conversion_util
from utils import label_map_util, visualization_utils as vis_util

# Setting up labels of trained model
PATH_TO_LABELS = cfg.PRETRAINED_MODEL_LABELS
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def read_srt_file(srt_file):
    if not os.path.exists(srt_file):
        print(f"The srt file {srt_file} does not exist. Quitting...")
        sys.exit()

    print(f"Using SRT file at: {srt_file}")

    # Reading srt file
    with open(srt_file, 'r') as f:
        data = f.read()
    srt_generator = srt.parse(data)
    srt_data = list(srt_generator)
    print(f"Constructed generator with {len(srt_data)} entries")
    return srt_data

# returns absolute path of newly created target folder
def createTargetFolder(inputFile):
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # Saving files into folder 'inputFilename_%Y-%m-%d_%H:%M:%S'
    sourcePathAbs = os.path.abspath(inputFile)
    sourceFileHead, sourceFileTail = os.path.split(sourcePathAbs)
    outputPath = sourceFileTail + "_" + now
    targetFolder = sourceFileHead + "/" + outputPath

    try:
        os.mkdir(targetFolder)
        print("Target directory ", targetFolder, " created")
    except FileExistsError:
        print("Target directory ", targetFolder, " already exists...")

    return targetFolder

def pack_detections(box, scores, classes, video_height, video_width, threshold = 0.5):
    n_boxes, field_boxes = box.shape
    assert field_boxes == 4, "Error: Bounding boxes should have 4 coordinates, has " + str(field_boxes)

    n_scores = scores.shape[0]
    assert n_scores == n_boxes, "Error: " + str(n_scores) + " scores returned, should equal to " + str(n_boxes) + " boxes"

    n_classes = classes.shape[0]
    assert n_classes == n_scores, "Error: " + str(n_classes) + " classes returned, should equal to " + str(n_scores) + " boxes"

    # extract coordinates
    coord_boxes = []
    for i, b in enumerate(box):
        # no need to process boxes under given threshold
        if scores[i] < threshold:
            continue

        # transform relative values to pixel values
        ymin = int(b[0] * video_height)
        xmin = int(b[1] * video_width)
        ymax = int(b[2] * video_height)
        xmax = int(b[3] * video_width)

        class_id = int(classes[i])

        if ymin == 0 and xmin == 0 and ymax == 0 and xmax == 0:
            # images does not contain any more detections (rest is 0)
            break

        # ATTENTION: Watch order!
        coord_boxes.append([xmin, ymin, xmax, ymax, class_id, scores[i]])
    return coord_boxes

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

def load_model(model_dir):
    model_dir = pathlib.Path(model_dir) / cfg.SAVED_MODEL_SUBFOLDER
    print("Loading saved model at ", model_dir)
    model = tf.compat.v2.saved_model.load(export_dir=str(model_dir), tags=None)
    model = model.signatures['serving_default']
    return model


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

def run_single_inference(sess, return_tensors, image_tensor, image_np):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection, returns (boxes, scores, class, num)
    return sess.run(
        return_tensors,
        feed_dict={image_tensor: image_np_expanded})

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

            return_tensors = [detection_boxes, detection_scores, detection_classes, num_detections]

            for image_path in TEST_IMAGE_PATHS:
                _, image_name = os.path.split(image_path)

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = np.array(conversion_util.jpg_image_to_array(image_path))

                (boxes, scores, classes, num) = run_single_inference(sess, return_tensors, image_tensor, image_np)

                image_np = draw_bboxes_on_image(image_np, boxes, classes, scores, category_index)

                save_image(image_np, image_name)

def draw_bboxes_on_image(image_np, boxes, classes, scores, category_index):
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
    )
    return image_np

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

def load_model(model_dir):
    model_dir = pathlib.Path(model_dir) / cfg.SAVED_MODEL_SUBFOLDER
    print("Loading saved model at ", model_dir)
    model = tf.compat.v2.saved_model.load(export_dir=str(model_dir), tags=None)
    model = model.signatures['serving_default']
    return model


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

def run_single_inference(sess, return_tensors, image_tensor, image_np):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection, returns (boxes, scores, class, num)
    return sess.run(
        return_tensors,
        feed_dict={image_tensor: image_np_expanded})

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

            return_tensors = [detection_boxes, detection_scores, detection_classes, num_detections]

            for image_path in TEST_IMAGE_PATHS:
                _, image_name = os.path.split(image_path)

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = np.array(conversion_util.jpg_image_to_array(image_path))

                (boxes, scores, classes, num) = run_single_inference(sess, return_tensors, image_tensor, image_np)

                image_np = draw_bboxes_on_image(image_np, boxes, classes, scores, category_index)

                save_image(image_np, image_name)