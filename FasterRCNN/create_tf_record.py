"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python create_tf_record.py --input_txt=dataset_train.txt  --output_tfrecord=train.record

  # Create test data:
  python create_tf_record.py -input_txt=dataset_test.txt  --output_tfrecord=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import tensorflow as tf

from core.config import cfg

from PIL import Image
from utils import dataset_util
import argparse
from tqdm import tqdm

# Will contain an array of class labels definied by cfg.FASTERRCNN.CLASSES
classLabels = []

"""
    image_path : String, containing the path to the image
    observations: List of observations for given image. One observation is [xmin, xmax, ymin, ymax, class-id]
"""
def create_tf_example(image_path, observations):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    _, image_name = os.path.split(image_path)
    filename = image_name.encode('utf-8')

    image_format = str.encode(cfg.IMAGES_TYPE)
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for ob in observations:
        xmins.append(float(ob[0]) / width)
        xmaxs.append(float(ob[1]) / width)
        ymins.append(float(ob[2]) / height)
        ymaxs.append(float(ob[3]) / height)

        labelID = int(ob[4])
        classes.append(labelID)
        classes_text.append(classLabels[labelID].encode('utf8'))


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/channels': dataset_util.int64_feature(3),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def batch_tf_record_by_file(annotation_filepath, writer):
    annotations = []
    with tf.io.gfile.GFile(annotation_filepath) as f:
        annotations = [line.replace(",", " ").split() for line in f]

    lengthOfFixedElems = 1
    lengthOfObservation = 5
    
    # Each element in 'annotations' is the line of the imported annotation file
    # Now deconstruct those annotation for n observations per line
    for item in tqdm(annotations):
        filepath = item[0]
        observations = [item[1:6]]

        # if still observations are left, iterate through all of them
        obsCount = 1
        while(item[lengthOfFixedElems + lengthOfObservation * obsCount : ]):
            currentObservation = item[
                  lengthOfFixedElems + lengthOfObservation * obsCount :
                  lengthOfFixedElems + lengthOfObservation * (obsCount + 1)
                          ]
            observations.append(currentObservation)
            obsCount += 1

        tf_example = create_tf_example(filepath, observations)
        writer.write(tf_example.SerializeToString())

def main(annotation_filepath, output_file):
    with tf.io.gfile.GFile(cfg.FASTERRCNN.CLASSES ) as f:
        global classLabels
        classLabels = [line for line in f]

    writer = tf.io.TFRecordWriter(output_file)
    batch_tf_record_by_file(annotation_filepath, writer)
    writer.close()

    output_path = os.path.join(os.getcwd(), output_file)
    print('Successfully created the TFR'
          'ecords: {}'.format(output_path))


if __name__ == '__main__':
    # Taking command line arguments from users
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_txt', help='define the input dataset file (per line: path xmin ymin xmax ymax class-id)', type=str, required=False)
    parser.add_argument('-out', '--output_tfrecord', help='define the output file ', type=str, required=False)
    args = parser.parse_args()

    if(args.input_txt == None and args.output_tfrecord == None):
        if not os.path.exists("data"):
            os.makedirs("data")
        print("Begin processing training data...")
        main(cfg.TRAIN.ANNOT_PATH, cfg.TRAIN.RECORDS)
        print("Begin processing test data...")
        main(cfg.TEST.ANNOT_PATH, cfg.TEST.RECORDS)
    elif (args.input_txt == None or args.output_tfrecord == None):
        print("Illegal argument!")
        print("Please enter either both input dataset file and output file name or none (config file is then used).")
    else:
        main(args.input_txt, args.output_tfrecord)






