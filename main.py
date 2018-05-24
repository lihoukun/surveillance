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

from object_detection import image_util

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
    label = 0
    if args.image:
        for image_name in args.image.split(','):
            if os.path.isfile((image_name)):
                fimage = {}
                fimage['image_path'] = image_name
                fimage['id'] = label
                images.append(fimage)
                label += 1
    elif args.image_path:
        for image_name in sorted(os.listdir(args.image_path)):
            fimage = {}
            fimage['image_path'] = os.path.join(args.image_path, image_name)
            fimage['id'] = label
            images.append(fimage)
            label += 1
    elif args.video:
        try:
            if not os.path.isdir('input'):
                os.makedirs('input')
            cap = cv2.VideoCapture(args.video)
            while(1):
                ret, image = cap.read()
                if not ret: break
                cv2.imwrite('input/{}.jpg'.format(label), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                fimage = {}
                fimage['image_path'] = 'input/{}.jpg'.format(label)
                fimage['id'] = label
                images.append(fimage)
                label += 1
        except:
            print('capture video failed')
            exit(1)
    return images

def get_cfg(name):
    with open('models.yml', 'r') as f:
        cfg = yaml.load(f)
    return cfg[name]
    
def prepare_od_model():
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

    for i in range(num):
        cat = category_index[classes[i]]['name']
        if cat not in fimage:
            fimage[cat] = {}
        fimage[cat][len(fimage[cat])+1] = {'bbox': boxes[i].tolist(), 'score': float(scores[i])}
    print(fimage)

def object_detect(images, graph):
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
        for fimage in images:
            image_np = cv2.imread(fimage['image_path'])
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            try:
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(image_np, 0)}
                )

                #image_util.visualize_boxes_and_labels_on_image_array(
                #    image_np,
                #    np.squeeze(boxes),
                #    np.squeeze(classes).astype(np.int32),
                #    np.squeeze(scores),
                #    category_index,
                #    use_normalized_coordinates=True
                #)
                update_fimage_from_od(fimage, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), int(num[0]))

                #cv2.imwrite('output/{}.jpg'.format(count), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                count += 1
            except:
                print('fail to handle image {}'.format(fimage.image_path))
                continue
        end = datetime.datetime.now()
        print('total time: {}'.format(end-start))
        print('total image: {}'.format(count-1))

def save_json(images):
    with open('result.yml', 'w+') as f:
        yaml.dump(images, f, default_flow_style=False)

def main():
    args = parse_args()
    images = prepare_images(args)
    if 'od' in args.stages:
        detection_graph = prepare_od_model()
        object_detect(images, detection_graph)
        save_json(images)
    if args.video and args.output_type == 'bbox':
        os.system(r'ffmpeg -r 24 -i output/%d.jpg -vcodec mpeg4 -y video.mp4')

if __name__ == '__main__':
    main()
