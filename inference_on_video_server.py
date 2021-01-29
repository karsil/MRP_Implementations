#! /usr/bin/env python
# coding=utf-8

import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys
import argparse
from datetime import datetime
import srt
from tqdm import tqdm

from core.config import cfg
import utils.detect_util as detect_util
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
        video_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Begin processing of video with {maxFrames} frames...")

        jumpedFrames = 0
        # frames
        currFrame = jumpedFrames

        # progressbar
        pbar = tqdm(total=maxFrames)
        vid.set(cv2.CAP_PROP_POS_FRAMES, currFrame)

        while (vid.isOpened()):
            if currFrame == jumpedFrames:
                pbar.update(jumpedFrames)

            ret, frame = vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #image = Image.fromarray(frame)
            else:
                if (currFrame > jumpedFrames):
                    # something has been processed earlier
                    print("Done. Quitting...")
                    break
                else:
                    raise ValueError("Error while reading!", frame)

            (boxes, scores, classes, num) = detect_util.run_single_inference(sess, return_tensors, image_tensor, frame)
          
            score_threshold = 0.5
            final_score = np.squeeze(scores)
            count = 0

            for i in range(100):
                if scores is None or final_score[i] >= score_threshold:
                    count = count + 1
            
            if count > 0:
                # print(f"Found {count} detections in frame {currFrame} ")
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                )

                # saving as image
                image = Image.fromarray(frame)
                exportName = srtData[currFrame].content
                filepath = targetFolder + "/" + exportName + ".jpg"
                
                image.save(filepath)

                box = np.squeeze(boxes)
                
                _, fields = box.shape

                assert fields is 4, "Error: Bounding boxes should have 4 coordinates, has " + fields

                score_arr = np.squeeze(scores)
                classes_arr = np.squeeze(classes)

                coord_boxes = []
                for i, b in enumerate(box):
                    
                    score_of_detection = score_arr[i]
                    if score_of_detection < score_threshold:
                        continue 

                    ymin = int(b[0] * video_height)
                    xmin = int(b[1] * video_width)
                    ymax = int(b[2] * video_height)
                    xmax = int(b[3] * video_width)

                    class_id = int(classes_arr[i])

                    if ymin == 0 and xmin == 0 and ymax == 0 and xmax == 0:
                        # images does not contain any more detections
                        print(f"Breaking after {i} detections")
                        break
                    
                    # ATTENTION: Watch order!
                    coord_boxes.append([xmin, ymin, xmax, ymax, class_id])


                # Save logfile for image
                # Format: TopleftX, TopleftY, BottomRightX, BottomRightY, Class ID
                filepathLog = targetFolder + "/" + exportName + ".txt"
                with open(filepathLog, "w") as logfile:
                    for coords in coord_boxes:
                        (xmin, ymin, xmax, ymax, score) = coords
                        logfile.write(str(xmin) + ", " + str(ymin) + ", " + str(xmax) + ", " + str(ymax) + ", " + str(class_id) + "\n")

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
    parser.add_argument('-in', '--input_video', help='Provide the path to a file with data')

    args = parser.parse_args()

    inputFile = args.input_video

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
