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
import srt
from tqdm import tqdm

from core.config import cfg
from utils import detect_util
from utils import label_map_util
from utils import session_util

def run(detection_graph, inputFile, targetFolder, log_output = False, skip_frames = 0, srt_data = None,):
    # Setting up labels of trained model
    PATH_TO_LABELS = cfg.FASTERRCNN.LABELMAP
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    with tf.compat.v1.Session(graph=detection_graph) as sess:

        image_tensor = session_util.get_image_tensor(detection_graph)
        return_tensors = session_util.get_detection_tensors(detection_graph)

        print("Reading video: " + inputFile)
        vid = cv2.VideoCapture(inputFile)

        maxFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Begin processing of video with {maxFrames} frames...")

        # Set the frame counter
        iteration = skip_frames
        if iteration > 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, iteration)

        # progressbar
        pbar = tqdm(total=maxFrames)

        while (vid.isOpened()):

            # Update progressbar to skipped frame on first iteration
            if iteration == skip_frames:
                pbar.update(iteration)

            ret, frame = vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                if (iteration > 0):
                    # something has been processed earlier
                    print("Done. Quitting...")
                    break
                else:
                    raise ValueError("Error while reading!", frame)

            (boxes, scores, classes, num) = detect_util.run_single_inference(sess, return_tensors, image_tensor, frame)

            # check, if detections has been found
            SCORE_THRESHOLD = 0.5

            scores = np.squeeze(scores)
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)

            detections_in_image = 0
            # Count detections in image
            for score in scores:
                if scores is None or score >= SCORE_THRESHOLD:
                    detections_in_image += 1

            if detections_in_image > 0:
                image = detect_util.draw_bboxes(image, boxes, classes, scores, category_index)
                # saving as image
                image = Image.fromarray(image)
                # check if name is given
                if srt_data:
                    exportName = srt_data[iteration].content
                else:
                    # else use current frame index
                    exportName = str(iteration)

                filepath = targetFolder + "/" + exportName + ".jpg"
                image.save(filepath)

                if log_output:
                    filepath_log = targetFolder + "/" + exportName + ".txt"

                    detections = detect_util.pack_detections(boxes, scores, classes, video_height, video_width, SCORE_THRESHOLD)

                    # Save logfile for image
                    # Format: TopleftX, TopleftY, BottomRightX, BottomRightY, Class ID
                    with open(filepath_log, "w") as logfile:
                        for detection in detections:
                            (xmin, ymin, xmax, ymax, class_id, score) = detection
                            logfile.write(
                                str(xmin) + ", " + str(ymin) + ", " + str(xmax) + ", " + str(ymax) + ", " + str(
                                    class_id) + "\n")

            pbar.update(1)
            iteration = iteration + 1

        vid.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-video', '--input_video', help='Provide the path to a file with data')
    parser.add_argument('-srt', '--input_srt', help='Provide the path to a srt file for the given video')
    parser.add_argument('-skip', '--skip_frames', type=int, default=int(0),help='Provide an amount of frames, which shall be skipped. The processing will began from this frame on.')
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    inputFile = args.input_video

    if not os.path.exists(inputFile):
        print(f"The video file {inputFile} does not exist. Quitting...")
        sys.exit()

    srt_data = detect_util.read_srt_file(args.input_srt)

    targetFolder = detect_util.createTargetFolder(inputFile)

    detection_graph = detect_util.import_graph(cfg.TRAINED_MODEL_PATH)

    run(detection_graph, inputFile, targetFolder, args.log, args.skip_frames, srt_data)

    print(f"Done! files have been saved to folder ", targetFolder)

if __name__ == "__main__":
    main()