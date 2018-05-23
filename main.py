import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import argparse
import datetime
import json
import cv2
import yaml

import image_util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='input images seperated by commar')
    parser.add_argument('--image_path', help='input image path')
    parser.add_argument('--video', help='input local video file')
    parser.add_argument('--output_type', choices=['bbox', 'warp', 'chop'], default='bbox', help='output type')
    parser.add_argument('--width', help='ouput image width', type=int)
    parser.add_argument('--height', help='ouput image height', type=int)

    parser.add_argument('--stages', choices=['od', 'fd', 'od,fd'], default='od', help='stages')
    args = parser.parse_args()
    return args

def prepare_images(args):
    images = []
    if args.image:
        for image_name in args.image.split(','):
            if os.path.isfile((image_name)):
                image_np = cv2.imread(image_name)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                images.append(image_np)
    elif args.image_path:
        for image_name in os.listdir(args.image_path):
            image_np = cv2.imread(os.path.join(args.image_path, image_name))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            images.append(image_np)
    elif args.video:
        try:
            cap = cv2.VideoCapture(args.video)
            while(1):
                ret, image = cap.read()
                if not ret: break
                images.append(image)
        except:
            print('capture video failed')
    return images

def prepare_od_model():
    with open('models.yml', 'r') as f:
        cfg = yaml.load(f)

    od_cfg = cfg['object_detection']
    if not os.path.isfile(od_cfg['pb_path']):
        opener = urllib.request.URLopener()
        opener.retrieve(od_cfg['download_path'])
        tar_file = tarfile.open('{}.tar.gz'.format(od_cfg['model_name']))
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, './')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(od_cfg['pb_path'], 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


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
                    use_normalized_coordinates=True
                )

                cv2.imwrite('output/{}.jpg'.format(count), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                count += 1
            except:
                print('fail to handle image {}'.format(image_name))
                continue
        end = datetime.datetime.now()
        print('total time: {}'.format(end-start))
        print('total image: {}'.format(count-1))

def main():
    args = parse_args()
    images = prepare_images(args)
    if 'od' in args.stages:
        detection_graph = prepare_od_model()
        images = detect_images(images, detection_graph)
    if args.video and args.output_type == 'bbox':
        os.system(r'ffmpeg -r 24 -i output/%d.jpg -vcodec mpeg4 -y video.mp4')

if __name__ == '__main__':
    main()
