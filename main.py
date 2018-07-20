from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import yaml
import shutil
import re

import image_utils
import object_detection
import face_detection
import person_reid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='input images seperated by commar')
    parser.add_argument('--image_path', help='input image path')
    parser.add_argument('--video', help='input local video file')
    parser.add_argument('--video_frames', help='max video frames to get', type=int)
    parser.add_argument('--frame_dir', help='per frame image dir for video')
    parser.add_argument('--person_dir', help='per person image dir for video')
    parser.add_argument('--face_dir', help='per face image dir for video')
    parser.add_argument('--yaml_dir', help='per image yaml dir')
    parser.add_argument('--namelist', help='yaml file to map person name and image')
    parser.add_argument('--distance', help='euclidean distance to treat as same person', type=float)
    parser.add_argument('--video_out', help='output video with bbox', action='store_true')

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
        image_utils.save_image_from_video(args.frame_dir, args.video, args.video_frames)
        for image_name in sorted(os.listdir(args.frame_dir)):
            images[os.path.splitext(image_name)[0]] = {'image_path': os.path.join(args.frame_dir, image_name)}
    return images

def save_yaml(images, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    keys = sorted(images.keys())
    print('Total {} yaml files to save'.format(len(keys)))
    for fid in keys:
        v = images[fid]
        filename = os.path.join(output_dir, '{}.yml'.format(fid))
        print('\rSaving yaml file {}'.format(filename), flush=True, end='')
        with open(filename, 'w+') as f:
            yaml.dump(v, f, default_flow_style=False)
    print('')

def load_yaml(yaml_dir):
    data = {}
    filenames = sorted(os.listdir(yaml_dir))
    print('Total {} yaml files to load'.format(len(filenames)))
    for filename in filenames:
        print('\rLoading yaml file {}'.format(filename), flush=True, end='')
        fid = filename.split('.')[0]
        with open(os.path.join(yaml_dir, filename), 'r') as f:
            data[fid] = yaml.load(f)
    print('')
    return data

def save_person(images, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print('')
    for fid in sorted(images.keys()):
        print('\rSaving person image for fid={}'.format(fid), flush=True, end='')
        fimage = images[fid]
        image_np = image_utils.read_image_to_np(fimage['image_path'])
        persons = fimage['detections']['person']
        if not persons: continue
        for pid in sorted(persons.keys()):
            v = persons[pid]
            image_name = '{}_person_{}.jpg'.format(fid, pid)
            image_chop = image_utils.read_image_from_np_with_box(v['bbox'], image_np)
            person_path = os.path.join(output_dir, image_name)
            image_utils.save_image_from_np(person_path, image_utils.resize_image_from_np(image_chop, 256, 128))
            v['image_path'] = person_path
    print('')

def load_name_vector(namelist, images):
    name_vectors = {}
    with open(namelist, 'r') as f:
        name_dict = yaml.load(f)
        for pname, fnames in name_dict.items():
            name_vectors[pname] = []
            for fname in fnames:
                m = re.search('(\d+)_person_(\d+)', fname)
                fid = str(m.group(1)).zfill(6)
                pid = int(m.group(2))
                name_vectors[pname].append(images[fid]['detections']['person'][pid]['embedding'])
    return name_vectors

def main():
    args = parse_args()
    if not args.yaml_dir:
        print('yaml_dir not defined!')
        exit(1)

    if args.video:
        images = prepare_images(args)
    else:
        images = load_yaml(args.yaml_dir)

    if  args.person_dir:
        detection_graph = object_detection.prepare_model()
        images = object_detection.detect(images, detection_graph)
        save_person(images, args.person_dir)
        save_yaml(images, args.yaml_dir)
        person_reid.embed(images)

    if args.face_dir:
        # pnet, rnet, onet = face_detection.prepare_model()
        # images = face_detection.detect(images, pnet, rnet, onet)
        pass

    if args.namelist:
        name_vectors = load_name_vector(args.namelist, images)
        distance = person_reid.get_distance(name_vectors)
        person_reid.reid(images, name_vectors, args.distance)
        save_yaml(images, args.yaml_dir)
        
    if args.video_out:
        image_dir = 'data/output/frames'
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        keys = sorted(images.keys())
        if args.distance:
            distance = args.distance
        elif not distance:
            distance = 12.0
        print('distance set to {}'.format(distance))
        print('')
        for i in range(len(keys)):
            fimage = images[keys[i]]
            print('\rSaving frame {}'.format(i), flush=True, end='')
            image_utils.save_image_from_fimage('{}/{}.jpg'.format(image_dir, i), fimage, distance)
        print('')
        os.system(r'ffmpeg -r 30 -i data/output/frames/%d.jpg -vcodec mpeg4 -y data/output/video.mp4')

if __name__ == '__main__':
    main()
