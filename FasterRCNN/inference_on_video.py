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
import detect_util
from utils import label_map_util, visualization_utils as vis_util
from utils import session_util

def run(detection_graph, inputFile, targetFolder, log_output = False, skip_frames = 0, srt_data = None,):
    # Setting up labels of trained model
    PATH_TO_LABELS = cfg.FASTERRCNN.LABELMAP
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    with tf.compat.v1.Session(graph=detection_graph) as sess:

        image_tensor = session_util.get_input_tensor(detection_graph)
        return_tensors = session_util.get_detection_tensors(detection_graph)

        print("Reading video: " + inputFile)
        vid = cv2.VideoCapture(inputFile)

        maxFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Begin processing of video with {maxFrames} frames...")

        # Set the frame counter
        current_frame = skip_frames
        if current_frame > 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # progressbar
        pbar = tqdm(total=maxFrames)

        while (vid.isOpened()):

            # Update progressbar to skipped frame on first iteration
            if current_frame == skip_frames:
                pbar.update(current_frame)

            ret, frame = vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                if (current_frame > 0):
                    # something has been processed earlier
                    print("Done. Quitting...")
                    break
                else:
                    raise ValueError("Error while reading!", frame)

            (boxes, scores, classes, num) = detect_util.run_single_inference(sess, return_tensors, image_tensor, frame)

            # check, if detections has been found
            score_threshold = 0.5

            scores = np.squeeze(scores)
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)

            frame_has_detection = False

            # check, if there are any detections at all above threshold
            for score in scores:
                if scores is None or score >= score_threshold:
                    frame_has_detection = True
                    break

            if frame_has_detection > 0:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    boxes,
                    classes.astype(np.int32),
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                )

                # saving as image
                image = Image.fromarray(image)
                # check if name is given
                if srt_data:
                    exportName = srt_data[current_frame].content
                else:
                    # use current frame index
                    exportName = str(current_frame)

                filepath = targetFolder + "/" + exportName + ".jpg"
                image.save(filepath)

                if log_output:
                    filepath_log = targetFolder + "/" + exportName + ".txt"

                    detections = pack_detections(boxes, scores, classes, video_height, video_width, score_threshold)

                    # Save logfile for image
                    # Format: TopleftX, TopleftY, BottomRightX, BottomRightY, Class ID
                    with open(filepath_log, "w") as logfile:
                        for detection in detections:
                            (xmin, ymin, xmax, ymax, class_id, score) = detection
                            logfile.write(
                                str(xmin) + ", " + str(ymin) + ", " + str(xmax) + ", " + str(ymax) + ", " + str(
                                    class_id) + "\n")

            pbar.update(1)
            current_frame = current_frame + 1

        vid.release()

def pack_detections(box, scores, classes, video_height, video_width, threshold = 0.5):
    n_boxes, field_boxes = box.shape
    assert field_boxes == 4, "Error: Bounding boxes should have 4 coordinates, has " + field_boxes

    n_scores, field_scores = scores.shape
    assert n_scores == n_boxes, "Error: " + n_scores + " scores returned, should equal to " + n_boxes + " boxes"

    n_classes, field_classes = classes.shape
    assert field_classes == 1, "Error: More than one label found"
    assert n_classes == n_scores, "Error: " + n_classes + " classes returned, should equal to " + n_scores + " boxes"

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
        coord_boxes.append([xmin, ymin, xmax, ymax, class_id, scores[i] ])
    return coord_boxes


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-video', '--input_video', help='Provide the path to a file with data')
    parser.add_argument('-srt', '--input_srt', help='Provide the path to a srt file for the given video')
    parser.add_argument('-skip', '--skip_frames', help='Provide an amount of frames, which shall be skipped. The processing will began from this frame on.')
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    inputFile = args.input_video

    if not os.path.exists(inputFile):
        print(f"The video file {inputFile} does not exist. Quitting...")
        sys.exit()

    srt_data = read_srt_file(args.input_srt)

    targetFolder = createTargetFolder(inputFile)

    detection_graph = detect_util.import_graph(cfg.TRAINED_MODEL_PATH)

    run(detection_graph, inputFile, targetFolder, args.log, args.skip_frames, srt_data)

    print(f"Done! files have been saved to folder ", targetFolder)

if __name__ == "__main__":
    main()