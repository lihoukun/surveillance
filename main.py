from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import yaml
import shutil

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
    parser.add_argument('--init_db', help='initialize unlabbbled data in sqlite', action='store_true')
    parser.add_argument('--frame_dir', help='per frame image dir for video')
    parser.add_argument('--person_dir', help='per person image dir for video')
    parser.add_argument('--face_dir', help='per face image dir for video')
    parser.add_argument('--cluster_dir', help='person image moved into cluster')
    parser.add_argument('--num_cluster', help='number of cluster for person reid', type=int)
    parser.add_argument('--yaml_dir', help='per image yaml dir')

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

def save_person_by_cluster(images, output_dir):

    print('')
    for fid in sorted(images.keys()):
        fimage = images[fid]
        persons = fimage['detections']['person']
        if not persons: continue
        for pid in sorted(persons.keys()):
            v = persons[pid]
            src_path = v['image_path']
            dst_base = os.path.join(output_dir, str(v['cluster_index']))
            if not os.path.isdir(dst_base):
                os.makedirs(dst_base)
            image_name = '{}_person_{}.jpg'.format(fid, pid)
            dst_path = os.path.join(dst_base, image_name)
            shutil.copyfile(src_path, dst_path)
            print('\rCoping person image for pid={}, fid={}'.format(pid, fid), flush=True, end='')
    print('')

def no_use():
    if args.vis_bbox:
        print('saving image with bbox')
        count  = 1
        for id, fimage in images.items():
            image_utils.save_image_from_fimage('{}/{}.jpg'.format(args.output_dir, id), fimage)
            if count % 100 == 0:
                print('{} images finished'.format(count))
            count += 1

    if args.video:
        os.system(r'ffmpeg -r 30 -i output/%d.jpg -vcodec mpeg4 -y video.mp4')


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

    if args.face_dir:
        # pnet, rnet, onet = face_detection.prepare_model()
        # images = face_detection.detect(images, pnet, rnet, onet)
        pass

    if args.num_cluster:
        person_reid.reid(images, args.num_cluster)
        if args.cluster_dir:
            save_person_by_cluster(images, args.cluster_dir)
        save_yaml(images, args.yaml_dir)

if __name__ == '__main__':
    main()
