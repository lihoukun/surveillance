import os
import tarfile
import six.moves.urllib as urllib
import tensorflow as tf
import numpy as np
import datetime
import cv2

import image_utils

def prepare_model():
    base_dir = 'models/object_detection'
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    #model_name = 'faster_rcnn_nas_coco_2018_01_28'
    pb_path = os.path.join(base_dir, model_name, 'frozen_inference_graph.pb')
    link = 'http://download.tensorflow.org/models/object_detection/{}.tar.gz'.format(model_name)
    tar_name = os.path.basename(link)
    tar_path = os.path.join(base_dir, tar_name)

    if not os.path.isfile(pb_path):
        if not os.path.isfile(tar_path):
            opener = urllib.request.URLopener()
            opener.retrieve(link, tar_path)
        print('tar file ready at {}'.format(tar_path))
        tar_file = tarfile.open(tar_path)
        for filename in tar_file.getmembers():
            if 'frozen_inference_graph.pb' in os.path.basename(filename.name):
                tar_file.extract(filename, base_dir)

    if not os.path.isfile(pb_path):
        print('no pb file')
        exit(1)

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
        print('')
        first_frame = None
        for fid, fimage in images.items():
            image = cv2.imread(fimage['image_path'])
            
            # blur the image to smooth noise
            frame = cv2.GaussianBlur(image, (21, 21), 0)
            # if the first frame is None, initialize it
            if first_frame is None:
                first_frame = frame
                continue
              
            # compute the absolute difference between the current frame and first frame
            frame_delta = cv2.absdiff(first_frame, frame)
            frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(frame_delta, 10, 255, cv2.THRESH_BINARY)[1]
            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=10)
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            total_boxes = []
            total_scores = []
            total_classes = []
            total_num = 0
            for cnt in cnts:
                if cv2.contourArea(cnt) < 50: # filter small areas of size less then 50 pxl
                    continue
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(cnt)
                # HARDCODE for demo
                if y < 330 or max(w,h)<30:
                    continue              
                image_crop = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                boxes, scores, classes, num = sess.run(
                      [detection_boxes, detection_scores, detection_classes, num_detections],
                      feed_dict={image_tensor: np.expand_dims(image_crop, 0)}
                )
                if int(num[0]) > 0:
                    total_boxes.extend(np.squeeze(boxes))
                    total_scores.extend(np.squeeze(scores))
                    total_classes.extend(np.squeeze(classes).astype(np.int32))
                    total_num += int(num[0])
            
            update_fimage_from_od(fimage, total_boxes, total_scores, total_classes, total_num)
            image_count += 1
            if image_count % 100 == 0:
                loop_end = datetime.datetime.now()
                print('\rProcessed to image {},  speed: {} image/second'.format(image_count, 
                        100 / (loop_end-loop_start).total_seconds()), flush=True, end='')
                loop_start = loop_end
            object_count += total_num
        end = datetime.datetime.now()
        print('')
        print('total object detection time: {}'.format(end - start))
        print('total detected objects: {}'.format(object_count))

    return images
