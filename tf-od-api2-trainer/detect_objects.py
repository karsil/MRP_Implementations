import os
import cv2
import time
import argparse
from pathlib import Path
from typing import List
import shutil
from abc import ABC
from tqdm import tqdm


from detector import DetectorTF2

UFO_CLASSES = ["fish_clupeidae", "jellyfish_aurelia", "fish_unspecified", "fish_cod", "fish_herring", "jellyfish_unspecified"]


class BBox(ABC):
	pass


class Detection(BBox):
	def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int, class_name: str, score: float):
			self.xmin = xmin,
			self.ymin = ymin
			self.xmax = xmax
			self.ymax = ymax
			self.class_name = class_name
			self.score = score
	
	def __str__(self):
		# fish_cod 0.05917061120271683 9 6 79 410
		# TODO Why the hell does this one not work?
		xmin = str(self.xmin).replace(",", '').replace("(", '').replace(")", '')
		return f"{self.class_name} {str(self.score)} {str(xmin)} {str(self.ymin)} {str(self.xmax)} {str(self.ymax)}"


class UFOAnnotation(BBox):
	def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int, class_id: int):
		self.xmin = xmin,
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.class_id = class_id
		self.class_name = UFO_CLASSES[class_id]

		#print(f"Create: {str(self)} -> {self.class_name}")

	def __str__(self):
		# goal: e.g. "fish_cod 109 0 195 69"
		# TODO Why the hell does this one not work?
		xmin = str(self.xmin).replace(",", '').replace("(", '').replace(")", '')
		return f"{self.class_name} {xmin} {str(self.ymin)} {str(self.xmax)} {str(self.ymax)}"


class Sample:
	def __init__(self, path: str, objects: List[BBox]):
		self.path = path
		self.objects = objects

	def save_in_folder(self, dir_path: Path):
		dst = dir_path.joinpath(self.path).with_suffix('.txt')
		#print(f"Store to {dst}")
		dst.touch()
		m = "\n".join(list(map(str, self.objects)))
		dst.write_text(m)


def parse_and_store_groundtruth(target_dir: Path, image_path: Path, annotations: List[UFOAnnotation]) -> None:
	sample = Sample(image_path.name, annotations)
	sample.save_in_folder(target_dir)


def parse_from_yolo(line: str) -> (Path, List[UFOAnnotation]):
	data = line.split()
	filepath = Path(data[0])
	annots = []
	for gt in data[1:]:
		gt = gt.split(",")
		gt_int = list(map(int, gt))
		annots.append(UFOAnnotation(gt_int[0], gt_int[1], gt_int[2], gt_int[3], gt_int[4]))
	return filepath, annots


def validate(detector: DetectorTF2, dataset_file: str, output_dir: str, save_output: bool = False):
	with open(dataset_file, 'r') as f:
		lines = f.readlines()
	
	dst = Path(output_dir)
	if dst.is_dir():
		shutil.rmtree(dst)
	dst.mkdir()

	gt_dir = dst.joinpath("groundtruth")
	gt_dir.mkdir()

	detect_dir = dst.joinpath("detection_results")
	detect_dir.mkdir()

	if save_output:
		bbox_dir = dst.joinpath("bbox_images")
		bbox_dir.mkdir()

	for l in tqdm(lines):
		filepath, annots = parse_from_yolo(line=l)
		assert filepath.is_file()

		# handle groundtruth
		parse_and_store_groundtruth(target_dir=gt_dir, image_path=filepath, annotations=annots)

		# handle inference
		img = cv2.imread(str(filepath))
		det_boxes = detector.DetectFromImage(img)
		img = detector.DisplayDetections(img, det_boxes)

		bboxes = [
			Detection(xmin=b[0], ymin=b[1], xmax=b[2], ymax=b[3], class_name=b[4], score=b[5]) for b in det_boxes
		]
		sample = Sample(filepath.name, bboxes)
		sample.save_in_folder(detect_dir)
		if save_output:
			img_out = os.path.join(str(bbox_dir), filepath.name)
			cv2.imwrite(img_out, img)


def eval_ufo():
	# instance of the class DetectorTF2
	model_path = "/home/jsteeg/Tensorflow/tf2-object-detection-api-tutorial/out/exported_model/saved_model"
	dataset_file = "/home/jsteeg/ufo_data/yolo_no_crop_vc/test.txt"
	output_dir = "tmp_validate_results"
	label_map = "/home/jsteeg/ufo_data/yolo_no_crop_vc/label_map.txt"
	save_output = True
	id_list = None
	threshold = 0.05
	detector = DetectorTF2(model_path, label_map, class_id=id_list, threshold=threshold)
	validate(detector=detector, dataset_file=dataset_file, output_dir=output_dir, save_output=save_output)



def DetectFromVideo(detector, Video_path, save_output=False, output_dir='output/'):

	cap = cv2.VideoCapture(Video_path)
	if save_output:
		output_path = os.path.join(output_dir, 'detection_'+ Video_path.split("/")[-1])
		frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))

	while (cap.isOpened()):
		ret, img = cap.read()
		if not ret: break

		timestamp1 = time.time()
		det_boxes = detector.DetectFromImage(img)
		elapsed_time = round((time.time() - timestamp1) * 1000) #ms
		img = detector.DisplayDetections(img, det_boxes, det_time=elapsed_time)

		#cv2.imshow('TF2 Detection', img)
		#if cv2.waitKey(1) == 27: break

		print("fuck")
		if save_output:
			out.write(img)

	cap.release()
	if save_output:
		out.release()


def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):

	for file in os.scandir(images_dir):
		if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
			image_path = os.path.join(images_dir, file.name)
			print(image_path)
			img = cv2.imread(image_path)
			det_boxes = detector.DetectFromImage(img)
			img = detector.DisplayDetections(img, det_boxes)

			#cv2.imshow('TF2 Detection', img)
			#cv2.waitKey(0)

			if save_output:
				img_out = os.path.join(output_dir, file.name)
				cv2.imwrite(img_out, img)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
	parser.add_argument('--model_path', help='Path to frozen detection model',
						default='models/efficientdet_d0_coco17_tpu-32/saved_model')
	parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file',
	                    default='models/mscoco_label_map.pbtxt')
	parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ","',
	                    type=str, default=None) # example input "1,3" to detect person and car
	parser.add_argument('--threshold', help='Detection Threshold', type=float, default=0.4)
	parser.add_argument('--images_dir', help='Directory to input images)', default='data/samples/images/')
	parser.add_argument('--video_path', help='Path to input video)', default='data/samples/pedestrian_test.mp4')
	parser.add_argument('--output_directory', help='Path to output images and video', default='data/samples/output')
	parser.add_argument('--video_input', help='Flag for video input, default: False', action='store_true')  # default is false
	parser.add_argument('--save_output', help='Flag for save images and video with detections visualized, default: False',
	                    action='store_true')  # default is false
	args = parser.parse_args()

	id_list = None
	if args.class_ids is not None:
		id_list = [int(item) for item in args.class_ids.split(',')]

	if args.save_output:
		if not os.path.exists(args.output_directory):
			os.makedirs(args.output_directory)

	# instance of the class DetectorTF2
	detector = DetectorTF2(args.model_path, args.path_to_labelmap, class_id=id_list, threshold=args.threshold)

	if args.video_input:
		DetectFromVideo(detector, args.video_path, save_output=args.save_output, output_dir=args.output_directory)
	else:
		DetectImagesFromFolder(detector, args.images_dir, save_output=args.save_output, output_dir=args.output_directory)

	print("Done ...")
	#cv2.destroyAllWindows()
