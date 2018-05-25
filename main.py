from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import argparse
import yaml

import image_utils
import object_detection
import face_detection

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='input images seperated by commar')
    parser.add_argument('--image_path', help='input image path')
    parser.add_argument('--video', help='input local video file')
    parser.add_argument('--vis_bbox', help='visualize original image with bounding box', action='store_true')
    parser.add_argument('--chop_obj', help='chop out object', action='store_true')
    parser.add_argument('--warp_face', help='warp out face', action='store_true')

    parser.add_argument('--stages', choices=['od', 'fd', 'od,fd'], default='od', help='stages')
    args = parser.parse_args()
    return args

def prepare_images(args):
    images = {}
    if args.image:
        for image_name in args.image.split(','):
            if os.path.isfile((image_name)):
                images[len(images)+1] = {'image_path': image_name}
    elif args.image_path:
        for image_name in sorted(os.listdir(args.image_path)):
            images[len(images)+1] = {'image_path': os.path.join(args.image_path, image_name)}
    elif args.video:
        image_utils.save_image_from_video('input', args.video)
        for image_name in sorted(os.listdir('input')):
            images[len(images)+1] = {'image_path': os.path.join('input', image_name)}
    return images

def save_json(images):
    with open('result.yml', 'w+') as f:
        yaml.dump(images, f, default_flow_style=False)

def save_image(images, args):
    if not os.path.isdir('output'):
        os.makedirs('output')

    for id, fimage in images.items():
        if args.vis_bbox:
            image_utils.save_image_from_fimage('ouput/{}.jpg'.format(id), fimage)
            if args.video:
                os.system(r'ffmpeg -r 24 -i output/%d.jpg -vcodec mpeg4 -y video.mp4')
        if args.chop_obj or args.warp_face:
            image_np = image_utils.read_image_to_np(fimage['image_path'])
            for cat, v1 in fimage['detections'].items():
                for pid, v2 in v1.items():
                    if args.chop_obj:
                        image_name = '{}_{}_{}.jpg'.format(id, cat, pid)
                        image_chop = image_utils.read_image_from_np_with_box(v2['bbox'], image_np)
                        image_utils.save_image_from_np(os.path.join('output', image_name), image_chop)
                    if args.warp_face and 'face' in v2:
                        image_name = '{0}_{1}_{2}_{2}.jpg'.format(id, cat, pid)
                        image_chop = image_utils.read_image_from_np_with_box(v2['face']['bbox'], image_np)
                        image_utils.save_image_from_np(os.path.join('output', image_name), image_utils.resize_image_from_np(image_chop))


def main():
    args = parse_args()
    images = prepare_images(args)
    if 'od' in args.stages:
        detection_graph = object_detection.prepare_model()
        images = object_detection.detect(images, detection_graph)
    if 'fd' in args.stages:
        pnet, rnet, onet = face_detection.prepare_model()
        images = face_detection.detect(images, pnet, rnet, onet)
    save_json(images)
    save_image(images, args)


if __name__ == '__main__':
    main()
