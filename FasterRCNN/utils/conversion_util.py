from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape
    (width, height, channels)
    """
    with Image.open(image_path) as image:
        (im_width, im_height) = image.size
        im_arr = np.frombuffer(image.tobytes(), dtype=np.uint8)
        if image.mode is 'L':
            im_arr = greyscale_array_to_rgb_array(im_arr)
        im_arr = im_arr.reshape((im_height, im_width, 3))
    return im_arr

def greyscale_array_to_rgb_array(im_arr):
    """
    Stack greyscale image three times to get a RGB image
    """
    return np.stack((im_arr,) * 3, axis=-1)

if __name__ == '__main__':
    # Taking command line arguments from users
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_folder', help='define the input folder', type=str, required=True)
    parser.add_argument('-out', '--output_folder', help='define the output folder', type=str, required=True)
    args = parser.parse_args()

    outputFolder = os.path.join(os.getcwd(), args.output_folder)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    inputFolder = os.path.join(os.getcwd(), args.input_folder)
    inputFiles = os.listdir(args.input_folder)

    for filename in tqdm(inputFiles):
        file = (os.path.join(inputFolder, filename))
        image = jpg_image_to_array(file)

        img = Image.fromarray(image)

        _, image_name = os.path.split(args.input_folder)
        outputPath = os.path.join(outputFolder, filename)
        img.save(outputPath)

    print("Conversion to " + outputFolder + " is done")