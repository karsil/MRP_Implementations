#! /usr/bin/env python
# coding=utf-8

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os
import sys
import argparse
from datetime import datetime
import srt
from tqdm import tqdm

from core.config import cfg
import detect_util
from utils import label_map_util, visualization_utils as vis_util

def run(detection_graph, inputFile, srtData, targetFolder):
    # Setting up labels of trained model
    PATH_TO_LABELS = cfg.PRETRAINED_MODEL_LABELS
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

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

        print("Reading video at " + inputFile)
        vid = cv2.VideoCapture(inputFile)

        maxFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Begin processing of video with {maxFrames} frames...")

        # frames
        currFrame = 0

        # progressbar
        pbar = tqdm(total=maxFrames)

        while (vid.isOpened()):
            ret, frame = vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                if (currFrame > 0):
                    # something has been processed earlier
                    print("Done. Quitting...")
                    break
                else:
                    raise ValueError("Error while reading!", frame)

            (boxes, scores, classes, num) = detect_util.run_single_inference(sess, return_tensors, image_tensor, image)

            print("boxes", boxes)
            print("scores", scores)
            print("classes", classes)
            print("num", num)

            if num > 0:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                )

                # saving as image
                image = Image.fromarray(image)
                exportName = srtData[currFrame].content
                filepath = targetFolder + "/" + exportName + ".jpg"

                # TODO: still YOLO stuff
                # Save logfile for image
                # Format: TopleftX, TopleftY, BottomRightX, BottomRightY, Class ID
                # filepathLog = targetFolder + "/" + exportName + ".txt"
                # with open(filepathLog, "w") as logfile:
                #     for i, bbox in enumerate(bboxes):
                #         coor = np.array(bbox[:4], dtype=np.int32)
                #         class_ind = int(bbox[5])
                #         logfile.write(
                #             str(coor[0]) + " " + str(coor[1]) + " " + str(coor[2]) + " " + str(coor[3]) + " " + str(
                #                 class_ind) + "\n")
                #
                # image.save(filepath)

            pbar.update(1)
            currFrame = currFrame + 1

        vid.release()

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

def getSrtFileByName(inputFile):
    base = os.path.splitext(inputFile)[0]
    srtFile = base + '.srt'
    print(srtFile)
    if not os.path.exists(srtFile):
        print(f"The srt file {inputFile} does not exist. Quitting...")
        sys.exit()

    # Reading srt file
    with open(srtFile, 'r') as f:
        data = f.read()
    srt_generator = srt.parse(data)
    srtData = list(srt_generator)
    print(f"Constructed generator with {len(srtData)} entries")
    return srtData

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Provide the path to a file with data')
    args = parser.parse_args()

    inputFile = args.file_path

    if not os.path.exists(inputFile):
        print(f"The video file {inputFile} does not exist. Quitting...")
        sys.exit()

    srtData = getSrtFileByName(inputFile)

    targetFolder = createTargetFolder(inputFile)

    detection_graph = detect_util.import_graph(cfg.TRAINED_MODEL_PATH)

    #detect_util.run_by_checkpoint_v1(graph)
    run(detection_graph, inputFile, srtData, targetFolder)

    print(f"Done! files have been saved to folder ", targetFolder)

if __name__ == "__main__":
    main()