from PIL import Image
import os
import numpy as np
import argparse

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape
    (width, height, channels)
    """
    with Image.open(image_path) as image:
        (im_width, im_height) = image.size
        im_arr = np.frombuffer(image.tobytes(), dtype=np.uint8)
        print(len(im_arr))
        if image.mode is 'L':
            im_arr = np.stack((im_arr,) * 3, axis=-1)
        im_arr = im_arr.reshape((im_height, im_width, 3))
    return im_arr

if __name__ == '__main__':
    # Taking command line arguments from users
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_image', help='define the input file', type=str, required=False)
    args = parser.parse_args()

    image = jpg_image_to_array(args.input_image)
    print(image)

    img = Image.fromarray(image)
    outputFolder = os.getcwd() + "/tmp/"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    outputPath = outputFolder + "out.jpg"
    img.save(outputPath)
    print("Result stored at " + outputPath)

