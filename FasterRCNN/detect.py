#! /usr/bin/env python
# Detector using the TensorFlow Object Detection API

from core.config import cfg
import detect_util

# def runByCheckpoint():
#     #with tf.Session() as sess:
#     with tf.compat.v1.Session() as sess:
#         model_path = '.' + cfg.DOWNLOADED_MODELS_FOLDER + cfg.PRETRAINED_MODEL_NAME + "/model.ckpt"
#         saver = tf.compat.v1.train.import_meta_graph(model_path + ".meta")
#         #saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./'))
#         saver.restore(sess, model_path)
#
#         PATH_TO_TEST_IMAGES_DIR = pathlib.Path(cfg.IMAGES_PATH)
#         TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*." + cfg.IMAGES_TYPE)))
#
#         image = np.array(Image.open(TEST_IMAGE_PATHS[0]))
#         input_tensor = tf.convert_to_tensor(image)
#
#         graph = tf.compat.v1.get_default_graph()
#         bboxes = graph.get_tensor_by_name("prefix/detection_boxes:0")
#         scores = graph.get_tensor_by_name("prefix/detection_scores:0")
#         detNum = graph.get_tensor_by_name("prefix/num_detections:0")
#         detClasses = graph.get_tensor_by_name("prefix/detection_classes:0")
#
#         print(bboxes)
#         #print(scores)
#         #print(detNum)
#         #print(detClasses)
#
#         bboxes = sess.run(bboxes,  feed_dict={
#                 "prefix/image_tensor:0": input_tensor}
#         )
#         print(bboxes)
        #print(scores)
        #print(detNum)
        #print(detClasses)


        # for image_path in TEST_IMAGE_PATHS:
        #     head, image_name = os.path.split(image_path)
        #     #processedImage = run_and_draw_bboxes(detection_model, image_path)
        #
        #     image = np.array(Image.open(image_path))
        #     input_tensor = tf.convert_to_tensor(image)
        #     output_dict = sess.run(input_tensor)
        #     print(output_dict)
        #     int(output_dict.pop('num_detections'))
        #     #processed_image = draw_bboxes(image_path, output_dict)
        #
        #     #save_image(processed_image, image_name)

if __name__ == '__main__':

    model_path = detect_util.download_model(cfg.PRETRAINED_MODEL_NAME)
    graph = detect_util.import_graph(model_path)
    detect_util.run_by_checkpoint_v1(graph)


