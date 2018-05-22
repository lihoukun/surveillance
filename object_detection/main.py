import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import argparse
import datetime
import json
import cv2

from PIL import Image

import image_util

# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_kitti_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'kitti_label_map.json')
NUM_CLASSES = 2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='input image, seperate by commar')
    parser.add_argument('--image_path', help='input image path')
    parser.add_argument('--video', help='local video file')
    parser.add_argument('--video_skip', help='frame skip interval', type=int)
    args = parser.parse_args()
    return args

def prepare_video(args):
    images = []
    if args.video:
        if not os.path.isfile(args.video):
            print('video not exist at {}'.format(args.video))
            exit(1)
        cap = cv2.VideoCapture(args.video)
        if args.video_skip:
            skip = args.video_skip
        else:
            skip = 0
        while(1):
            ret, image = cap.read()
            if not ret: break
            images.append(image)
            for _ in range(skip):
                ret, image = cap.read()

    return images


def prepare_images(args):
    images = []
    if args.image:
        for image_name in args.image.split(','):
            if os.path.isfile((image_name)):
                image = Image.open(image_name)
                image_np = load_image_into_numpy_array(image)
                images.append(image_np)
    elif args.image_path:
        for image_name in os.listdir(args.image_path):
            image = Image.open(os.path.join(args.image_path, image_name))
            image_np = load_image_into_numpy_array(image)
            images.append(image_np)
    return images


def prepare_model():
    if not os.path.isfile(PATH_TO_CKPT):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, './')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def detect_images(images, graph):
    with open(PATH_TO_LABELS) as f:
        cat_array = json.load(f)
    category_index = {}
    for index, item in enumerate(cat_array):
        category_index[index+1] = item

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
        for image_np in images:
            try:
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(image_np, 0)}
                )

                image_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates = True
                )

                image = Image.fromarray(image_np)
                image.save('output/{}.jpg'.format(count))
                count += 1
            except:
                print('fail to handle image {}'.format(image_name))
                continue
        end = datetime.datetime.now()
        print('total time: {}'.format(end-start))
        print('total image: {}'.format(count-1))

def main():
    detection_graph = prepare_model()
    args = parse_args()
    images = prepare_images(args)
    if images:
        detect_images(images, detection_graph)
    else:
        images = prepare_video(args)
        if images:
            detect_images(images, detection_graph)
    	    os.system(r'ffmpeg -r 24 -i output/%d.jpg -vcodec mpeg4 -y movie.mp4')

if __name__ == '__main__':
    main()
