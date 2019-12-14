#!/usr/bin/python
import random
import sys
import os
import argparse

trainRatio = 0.6

def isTrain():
    chance = random.random()
    if chance <= trainRatio:
        return True
    else:
        return False

def split(filename):
    testFileName = "test_" + filename
    trainFileName = "train_" + filename

    with open(filename, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)

    with open(trainFileName, 'w') as fTrain:
        with open(testFileName, 'w') as fTest:
            for line in lines:
                if (isTrain()):
                    fTrain.write(line)
                else:
                    fTest.write(line)

    return len(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', help='Provide the file name')
    parser.add_argument('training_ratio', nargs='?', help='Ratio of the training data, e.g. 0.6. Default is 0.6 training, 0.4 testing')
    args = parser.parse_args()

    inputFilename = args.file_name

    if not os.path.exists(inputFilename):
        print(f"The file {inputFilename} does not exist. Quitting...")
        sys.exit()

    if args.training_ratio is not None:
        ratio = float(args.training_ratio)
        if ratio < 0 or ratio > 1:
            print(f'A ratio of {ratio} is not legit. Please insert a non-negative number between 0 and 1. Quitting...')
            sys.exit()
        global trainRatio
        trainRatio = ratio

    print(f'Using a training/test ratio of {trainRatio}/{1.0 - trainRatio}.')   
    

    lineLength = split(inputFilename)

    print(f'Done after {lineLength} lines. Quitting...')

if __name__ == "__main__":
    main()