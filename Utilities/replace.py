#!/usr/bin/python
import argparse

def replace(source_file, target_file, source_string, target_string):
    """Replace all occurrences of source_string in source_file by target_string and save it to target_file.
    This can be used to replace relative paths to absolute paths and vice versa.
    """
    with open(source_file, "rt") as fin:
        with open(target_file, "wt") as fout:
            for line in fin:
                fout.write(line.replace(source_string, target_string))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--source_file', help='Provide the file path to the source file')
    parser.add_argument('-tf', '--target_file', help='Provide the file path to the target file which shall be written to')
    parser.add_argument('-ss', '--source_string', help='Provide the string which shall be replaced')
    parser.add_argument('-ts', '--target_string', help='Provide the string which shall be replaced by')
    args = parser.parse_args()

    replace(args.source_file, args.target_file, args.source_string, args.target_string)

