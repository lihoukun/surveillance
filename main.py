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
    parser.add_argument('--chop_person', help='chop out person', action='store_true')
    parser.add_argument('--warp_face', help='warp out face', action='store_true')
    parser.add_argument('--input_dir', help='input dir')
    parser.add_argument('--output_dir', help='output dir')

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
        image_utils.save_image_from_video(args.input_dir, args.video)
        for image_name in sorted(os.listdir(args.input_dir)):
            images[os.path.splitext(image_name)[0]] = {'image_path': os.path.join(args.input_dir, image_name)}
    return images

def save_json(images):
    with open('result.yml', 'w+') as f:
        yaml.dump(images, f, default_flow_style=False)

def save_image(images, args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.vis_bbox:
        print('saving image with bbox')
        count  = 1
        for id, fimage in images.items():
            image_utils.save_image_from_fimage('{}/{}.jpg'.format(args.output_dir, id), fimage)
            if count % 100 == 0:
                print('{} images finished'.format(count))
            count += 1

    #if args.video:
    #    os.system(r'ffmpeg -r 30 -i output/%d.jpg -vcodec mpeg4 -y video.mp4')

    if args.chop_person or args.warp_face:
        print('saving image part')
        count = 1
        for id, fimage in images.items():
            image_np = image_utils.read_image_to_np(fimage['image_path'])
            for cat, v1 in fimage['detections'].items():
                for pid, v2 in v1.items():
                    if args.chop_person:
                        image_name = '{}_person_{}.jpg'.format(id, pid)
                        image_chop = image_utils.read_image_from_np_with_box(v2['bbox'], image_np)
                        image_utils.save_image_from_np(os.path.join(args.output_dir, image_name), image_utils.resize_image_from_np(image_chop, 128, 64))
                    if args.warp_face and cat == 'face':
                        image_name = '{}_face_{}.jpg'.format(id, pid)
                        image_chop = image_utils.read_image_from_np_with_box(v2['bbox'], image_np)
                        image_utils.save_image_from_np(os.path.join(args.output_dir, image_name), image_utils.resize_image_from_np(image_chop, 112, 96))
            if count % 100 == 0:
                print('{} images finished'.format(count))
            count += 1


def main():
    args = parse_args()
    # raw
    images = prepare_images(args)
    # object detection
    detection_graph = object_detection.prepare_model()
    images = object_detection.detect(images, detection_graph)
    # face detection
    # pnet, rnet, onet = face_detection.prepare_model()
    # images = face_detection.detect(images, pnet, rnet, onet)
    # save result
    save_json(images)
    save_image(images, args)


if __name__ == '__main__':
    main()
