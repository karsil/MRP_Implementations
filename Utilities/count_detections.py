#!/usr/bin/python
import argparse

def count_detections_by_group(filepath, character, group_size = 4):
    """ Calculate the absolute frequency of detections inside an annotation file.
    A detection is definied as a group of characters, e.g. ','
    Example for two detections: PATH_IMAGE_1 xmin1, ymin1, xmax1, ymax1, class1 xmin2, ymin2, xmax2, ymax2
    """
    detection_counter = [0,0,0,0,0,0,0,0,0,0]

    with open(filepath, "rt") as dataset:
        for line in dataset:
            character_count_in_line = line.count(character)
            group_count = int(character_count_in_line / group_size)
            detection_counter[group_count] += 1

    return detection_counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--file_path', help='Provide the file path')
    parser.add_argument('-c', '--character', help='Provide the character to count for')
    args = parser.parse_args()

    detections = count_detections_by_group(args.file_path, args.character)

    for i, amount_of_detections in enumerate(detections):
        if amount_of_detections is 0:
            continue
        print(f"{amount_of_detections} entries contain {i} (bounding box/class) group\n")