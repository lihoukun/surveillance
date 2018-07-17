import os
import tarfile
import six.moves.urllib as urllib
import tensorflow as tf
import numpy as np
import datetime

import image_utils

def prepare_model():
    base_dir = 'models/object_detection'
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    pb_path = os.path.join(base_dir, 'frozen_inference_graph.pb')
    link = 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz'
    tar_name = os.path.basename(link)
    tar_path = os.path.join(base_dir, tar_name)

    if not os.path.isfile(pb_path):
        opener = urllib.request.URLopener()
        opener.retrieve(link, tar_path)
        tar_file = tarfile.open(tar_path)
        for file in tar_file.getmembers():
            if 'frozen_inference_graph.pb' in os.path.basename(file.name):
                tar_file.extract(file, './')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def update_fimage_from_od(fimage, boxes, scores, classes, num):
    fimage['detections'] = {}
    fimage['detections']['person'] = {}
    for i in range(num):
        if int(classes[i]) != 1: continue
        fimage['detections']['person'][len(fimage['detections']['person']) + 1] = {'bbox': boxes[i].tolist(),
                                                                         'score': float(scores[i])}
    return fimage


def detect(images, graph):
    with graph.as_default():
        sess = tf.Session(graph=graph)
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        start = datetime.datetime.now()
        loop_start = start
        image_count = 0
        object_count = 0
        for id, fimage in images.items():
            image_np = image_utils.read_image_to_np(fimage['image_path'])
            boxes, scores, classes, num = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(image_np, 0)}
            )
            update_fimage_from_od(fimage, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), int(num[0]))
            image_count += 1
            if image_count % 100 == 0:
                loop_end = datetime.datetime.now()
                print('Processed to image {},  speed: {} image/second'.format(image_count, 100 / (loop_end-loop_start).total_seconds()))
                loop_start = loop_end
            object_count += int(num[0])
        end = datetime.datetime.now()
        print('total object detection time: {}'.format(end - start))
        print('total detected objects: {}'.format(object_count))

    return images
