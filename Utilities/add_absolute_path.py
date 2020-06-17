#!/usr/bin/python
# Example:
# python add_absolute_path.py -i dataset_test.txt -f images/ -o tmp.txt

import argparse
import os
import re
from PIL import Image

TUPLE_LENGTH = 5
logger = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Provide the file path to the source file')
    parser.add_argument('-f', '--folder', help='Provide the directoy path to the files named in input-argument which shall be added')
    parser.add_argument('-o', '--output', help='Provide the file path to the target file which shall be written to')
    return parser.parse_args()

def read_file(file):
    with open(file, 'r') as f:
        txt = f.readlines()
        annots = [line.replace(",", " ").split(" ") for line in txt if len(line.strip().split()[0]) != 0]
    return annots

def write_annotations_to_file(annots, target_file):
    with open(target_file, "wt") as fout:
        for line in annots:
            fout.write(line[0] + " ")
            written_elems = 1
            while written_elems < len(line):
                fout.write(",".join(line[written_elems : written_elems + TUPLE_LENGTH]) + " ")
                written_elems = written_elems + TUPLE_LENGTH
            fout.write("\n")

    print("Wrote to: " + str(target_file))

def add_absolute_path_before_first_element(annots, folder):
    def assert_path_exists(path):
        assert os.path.isfile(path), "Error: File" + path + " does not exist"

    abs_path = str(os.path.abspath(folder))

    for i, annot in enumerate(annots):
        new_filepath = os.path.join(abs_path, annots[i][0])
        assert_path_exists(new_filepath)
        annots[i][0] = new_filepath
    return annots

def cleanup_values(annotations):
    def assert_bbox_range(detection, filename):
        xmin = int(float(detection[0]))
        ymin = int(float(detection[1]))
        xmax = int(float(detection[2]))
        ymax = int(float(detection[3]))

        # TODO: Use assert
        x_error_msg = "Dataset: xmin is larger than xmax: " + str(detection) + " for " + filename
        if not xmin < xmax:
            print(x_error_msg)
            logger.write(x_error_msg + "\n")

        y_error_msg = "Dataset: ymin is larger than ymax: " + str(detection) + " for " + filename
        if not ymin < ymax:
            print(y_error_msg)
            logger.write(y_error_msg + "\n")

        
        img = Image.open(filename)
        width, height = img.size
        msg = "Dataset: xmax " + str(xmax) + " higher than image width: " + str(width) + ": " + filename
        if not xmax <= width:
            print(msg)
            logger.write(msg + "\n")

        msg = "Dataset: ymax " + str(ymax) + " higher than image height: " + str(height) + ": " + filename
        if not ymax <= height:
            print(msg)
            logger.write(msg + "\n")

    
    def assert_dataset(annotations):
        for line in annotations:
            filename = line[0]
            read_elems = 1
            while read_elems < len(line):
                detection = line[read_elems : read_elems + TUPLE_LENGTH]
                assert_bbox_range(detection, filename)
                #print(",".join(detection))
                read_elems = read_elems + TUPLE_LENGTH

        # TODO: Use this, when assert is again implemented
        #print("Dataset has correct bounding boxes")

    def remove_linebreak(str):
        return str.replace("\n","")

    def remove_floating_point(str_num):
        return str(int(float(str_num)))

    def check_and_set_negative_values_to_one(str_num, filename = None):
        if int(str_num) < 0:
            postfix = filename if filename else ""
            error_msg = "Warning: Coordinate has negative value " + str_num + ", changing to 1..." + postfix
            print(error_msg)
            logger.write(error_msg + "\n")
            str_num = "1"
        return str_num

    assert_dataset(annotations)

    for i, annot in enumerate(annotations):
        for j, coord in enumerate(annot[1:]):
            annotations[i][j + 1] = remove_linebreak(annotations[i][j + 1])
            annotations[i][j + 1] = remove_floating_point(annotations[i][j + 1])
            annotations[i][j + 1] = check_and_set_negative_values_to_one(annotations[i][j + 1], annotations[i][0])

    return annotations

def create_log_path(relative_output_path):
    output = os.path.abspath(args.output)
    head, tail = os.path.split(args.output)
    return os.path.join(head, "log_" + tail)

def process(source_file, folder_path, target_file):
    
    annots = read_file(source_file)
    annots = add_absolute_path_before_first_element(annots, folder_path)
    annots = cleanup_values(annots)

    write_annotations_to_file(annots, target_file)

if __name__ == "__main__":
    args = parse_args()

    logger_path = create_log_path(args.output)
    logger = open(logger_path, "wt")

    process(args.input, args.folder, args.output)

    logger.close()
    print("Wrote logfile to " + logger_path)
    print("Please check logfile to verify result even if program returns no error")

