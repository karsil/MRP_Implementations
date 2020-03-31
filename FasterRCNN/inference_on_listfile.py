#! /usr/bin/env python
# coding=utf-8

import argparse
import numpy as np
import os
from core.config import cfg
from utils import detect_util
from utils import conversion_util
from utils import session_util
from utils import label_map_util
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def inference_on_image(sess, return_tensors, image_tensor, image_path, minimum_detections=1):
    image_np = np.array(conversion_util.jpg_image_to_array(image_path))
    (boxes, scores, classes, num) = detect_util.run_single_inference(
        sess,
        return_tensors,
        image_tensor,
        image_np
    )
    scores = np.squeeze(scores)
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)

    # check, if detections has been found
    SCORE_THRESHOLD = 0.5
    detections_in_image = 0
    # Count detections in image
    for score in scores:
        if scores is None or score >= SCORE_THRESHOLD:
            detections_in_image += 1

    if detections_in_image > minimum_detections:
        print(image_np.shape)
        (height, width, _) = image_np.shape
        return detect_util.pack_detections(
            boxes,
            scores,
            classes,
            height,
            width,
            SCORE_THRESHOLD
        )
    else:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_txt',
                        help='Provide the path to a file containing images which shall be processed')
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    annotation_file = args.input_txt
    store_logs = args.log

    targetFolder = detect_util.createTargetFolder(annotation_file)

    detection_graph = detect_util.import_graph(cfg.TRAINED_MODEL_PATH)

    annotations = conversion_util.read_annotations(annotation_file)

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = session_util.get_image_tensor(detection_graph)
            return_tensors = session_util.get_detection_tensors(detection_graph)

            filepath_log = targetFolder + "/" + "inference_results.txt"
            with open(filepath_log, "w") as logfile:
                for entry in tqdm(annotations):
                    image_path = entry[0]
                    detections = inference_on_image(sess, return_tensors, image_tensor, image_path, store_logs)

                    # Format (concatted detections), e.g.:
                    # IMG_PATH xmin1,ymin1,xmax1,ymax1,class1 xmin2,ymin2,xmax2,ymax2,class2
                    logfile.write(str(image_path) + " ")
                    for detection in detections:
                        (xmin, ymin, xmax, ymax, class_id, score) = detection
                        logfile.write(
                            str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + str(
                                class_id) + " ")
                    logfile.write("\n")

    print(f"Done! files have been saved to folder ", targetFolder)


if __name__ == "__main__":
    main()
