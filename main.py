import numpy as np
import tensorflow as tf
import os
import argparse
import yaml

import image_utils
import object_detection

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
        for image_name in sorted(os.listdir('input'):
            images[len(images)+1] = {'image_path': os.path.join('input', image_name)}
    return images

def save_json(images):
    with open('result.yml', 'w+') as f:
        yaml.dump(images, f, default_flow_style=False)

def save_image(images, method):
    for id, fimage in images.items():
        if method == 'bbox':
            image_utils.save_image_from_fimage('ouput/{}.jpg'.format(id), fimage)
        elif method == 'chop':
            base_dir = os.path.join('output', str(id))
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)

            image_np = image_utils.read_image_to_np(fimage['image_path'])
            for cat, v1 in fimage['detections'].items():
                for id, v2 in v1.items():
                    image_name = '{}_{}.jpg'.format(cat, id)
                    image_utils.save_image_from_np_with_box(os.path.join(base_dir, image_name), v2['bbox'], image_np)

def main():
    args = parse_args()
    images = prepare_images(args)
    if 'od' in args.stages:
        detection_graph = object_detection.prepare_model()
        object_detection.detect(images, detection_graph)
    save_json(images)
    save_image(images, args.output_type)

    if args.video and args.output_type == 'bbox':
        os.system(r'ffmpeg -r 24 -i output/%d.jpg -vcodec mpeg4 -y video.mp4')

if __name__ == '__main__':
    main()
