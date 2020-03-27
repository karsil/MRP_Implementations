
def get_detection_tensors(detection_graph):
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    return [detection_boxes, detection_scores, detection_classes, num_detections]

def get_image_tensor(detection_graph):
    # Definite input and output Tensors for detection_graph
    return detection_graph.get_tensor_by_name('image_tensor:0')