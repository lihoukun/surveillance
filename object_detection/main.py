import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import argparse
import datetime

from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_kitti_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'kitti_label_map.pbtxt')
NUM_CLASSES = 2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='input image, seperate by commar')
    parser.add_argument('--image_path', help='input image path')
    args = parser.parse_args()
    return args

def prepare_images(args):
    images = []
    if args.image:
        for image in args.image.split(','):
            if os.path.isfile((image)):
                images.append(image)
    elif args.image_path:
        for image_name in os.listdir(args.image_path):
            images.append(os.path.join(args.image_path, image_name))
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
    count = 0
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with graph.as_default():
        sess = tf.Session(graph=graph)
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        try:
            os.path.isdir('output')
        except:
            os.makedirs('output')

        start = datetime.datetime.now()
        for image_name in images:
            try:
                image = Image.open(image_name)
                image_np = load_image_into_numpy_array(image)
                print('a')
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(image, 0)}
                )

                print('b')
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates = True,
                    line_thickness = 8
                )

                print('c')
                image = Image.fromarray(image_np)
                image.save('output/new_{}'.format(os.path.basename(image_name)))
                print('d')
                count += 1
            except:
                print('fail to handle image {}'.format(image_name))
                continue
        end = datetime.datetime.now()
        print('total time: {}'.format(end-start))
        print('total image: {}'.format(count))

def main():
    detection_graph = prepare_model()
    images = prepare_images(parse_args())
    detect_images(images, detection_graph)

if __name__ == '__main__':
    main()
