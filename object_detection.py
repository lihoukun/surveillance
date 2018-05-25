import yaml
import os
import tarfile
import six.moves.urllib as urllib
import tensorflow as tf
import numpy as np
import json
import datetime

import image_utils

def get_cfg(name):
    with open('models.yml', 'r') as f:
        cfg = yaml.load(f)
    return cfg[name]


def prepare_model():
    cfg = get_cfg('object_detection')
    if not os.path.isfile(cfg['pb_path']):
        opener = urllib.request.URLopener()
        opener.retrieve(cfg['download_path'], '{}.tar.gz'.format(cfg['model_name']))
        tar_file = tarfile.open('{}.tar.gz'.format(cfg['model_name']))
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, './')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(cfg['pb_path'], 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def update_fimage_from_od(fimage, boxes, scores, classes, num):
    cfg = get_cfg('object_detection')
    with open(cfg['label_path']) as f:
        cat_array = json.load(f)
    category_index = {}
    for item in cat_array:
        category_index[item['id']] = item

    fimage['detections'] = {}
    for i in range(num):
        cat = category_index[classes[i]]['name']
        if cat not in fimage['detections']:
            fimage['detections'][cat] = {}
        fimage['detections'][cat][len(fimage['detections'][cat]) + 1] = {'bbox': boxes[i].tolist(),
                                                                         'score': float(scores[i])}
    return fimage


def detect(images, graph):
    cfg = get_cfg('object_detection')
    with open(cfg['label_path']) as f:
        cat_array = json.load(f)
    category_index = {}
    for item in cat_array:
        category_index[item['id']] = item

    with graph.as_default():
        sess = tf.Session(graph=graph)
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        if not os.path.isdir('output'):
            os.makedirs('output')

        start = datetime.datetime.now()
        count = 1
        for id, fimage in images.items():
            image_np = image_utils.read_image_to_np(fimage['image_path'])
            boxes, scores, classes, num = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(image_np, 0)}
            )
            update_fimage_from_od(fimage, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), int(num[0]))
            count += 1
        end = datetime.datetime.now()
        print('total time: {}'.format(end - start))
        print('total image: {}'.format(len(images)))

    return images
