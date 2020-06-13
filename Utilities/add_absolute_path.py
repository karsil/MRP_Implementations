#!/usr/bin/python
import argparse
import os
import re

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
    TUPLE_LENGTH = 5

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
    abs_path = str(os.path.abspath(folder))

    for i, annot in enumerate(annots):
        annots[i][0] =  abs_path + "/"  + annots[i][0]
    return annots

def cleanup_values(annotations):
    def remove_linebreak(str):
        return str.replace("\n","")

    def remove_comma(str_num):
        return str(int(float(str_num)))

    def check_and_set_negative_values_to_zero(str_num):
        if int(str_num) < 0:
            str_num = "0"
        return str_num

    for i, annot in enumerate(annotations):
        for j, coord in enumerate(annot[1:]):
            annotations[i][j + 1] = remove_linebreak(annotations[i][j + 1])
            annotations[i][j + 1] = remove_comma(annotations[i][j + 1])
            annotations[i][j + 1] = check_and_set_negative_values_to_zero(annotations[i][j + 1])

    return annotations

def process(source_file, folder_path, target_file):
    
    annots = read_file(source_file)
    annots = add_absolute_path_before_first_element(annots, folder_path)
    annots = cleanup_values(annots)

    write_annotations_to_file(annots, target_file)

if __name__ == "__main__":
    args = parse_args()

    process(args.input, args.folder, args.output)

