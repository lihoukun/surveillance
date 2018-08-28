from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import yaml

import image_utils
import object_detection
import face_detection
import person_reid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help='input local video file')
    parser.add_argument('--debug', help='debug mode, dump all needed files', action='store_true')
    # below options will be useful onl in debug mode
    parser.add_argument('--frame_dir', help='per frame image dir for video', default='data/frames')
    parser.add_argument('--dump_person', help='enable per person image dump', action='store_true')
    parser.add_argument('--person_dir', help='per person image dir for video', default='data/persons')
    parser.add_argument('--dump_face', help='enable per face image dump', action='store_true')
    parser.add_argument('--face_dir', help='per face image dir for video', default='data/faces')
    parser.add_argument('--yaml_dir', help='per image yaml dir', default='data/yamls')
    parser.add_argument('--discard_yml', help='yml file to store bbox to be discarded')
    parser.add_argument('--name_dir', help='path to trace images, with each subfolder as person name, and inside 256x128 image for that person')
    parser.add_argument('--dump_video', help='enable combined video dump', action='store_true')
    parser.add_argument('--output_dir', help='output dir', default='data/output')

    args = parser.parse_args()
    return args

def prepare_images(args):
    images = {}
    image_utils.save_image_from_video(args.frame_dir, args.video)
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

    ppaths = []
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
            ppaths.append(person_path)
    print('')
    return ppaths

def embed_name_vector(name_dir):
    colors = ['#ff0000', '#00ff00', '#0000ff', '#00aa55', '#5500aa', '#aa5500', '#0055aa', '#aa0055', '#55aa00']
    name_vectors = {}
    names = []
    ppaths = []

    for i, name in enumerate(os.listdir(name_dir)):
        name_path = os.path.join(name_dir, name)
        if not os.path.isdir(name_path): continue
        name_vectors[name] = {}
        name_vectors[name]['vectors'] = []
        name_vectors[name]['color'] = colors[i]

        for filename in os.listdir(name_path):
            image_path = os.path.join(name_path, filename)
            names.append(name)
            ppaths.append(image_path)

    vectors = person_reid.embed(ppaths)
    for i in range(len(names)):
        name = names[i]
        vector = vectors[i]
        name_vectors[name]['vectors'].append(vector)
    
    return name_vectors

def get_iou(bb1, bb2, method='square'):
    iou = 0.0
    if ',' in method:
        for meth in method.split(','):
            iou = max(iou, get_iou(bb1, bb2, method=meth))
        return iou

    ymin = max(bb1[0], bb2[0])
    xmin = max(bb1[1], bb2[1])
    ymax = min(bb1[2], bb2[2])
    xmax = min(bb1[3], bb2[3])

    if method == 'row':
        if xmax < xmin:
            return iou
        bb1_len = bb1[3] - bb1[1]
        bb2_len = bb2[3] - bb2[1]
        iou = (xmax-xmin) / float(bb1_len + bb2_len - (xmax-xmin))
    elif method == 'col':
        if ymax < ymin:
            return iou
        bb1_len = bb1[2] - bb1[0]
        bb2_len = bb2[2] - bb2[0]
        iou = (ymax-ymin) / float(bb1_len + bb2_len - (ymax-ymin))
    elif method == 'square':
        if xmax < xmin or ymax < ymin:
            return iou
        intersection_area = (xmax - xmin) * (ymax - ymin)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def discard_bbox(images, discard_yml):
    with open(discard_yml, 'r') as f:
        discard_bboxes = yaml.load(f)

    for fid in sorted(images.keys()):
        print('\rUpdating score for person fid {}'.format(fid), flush=True, end='')
        fimage = images[fid]
        persons = fimage['detections']['person']
        if not persons: continue
        for pid in sorted(persons.keys()):
            bbox = persons[pid]['bbox']
            score = persons[pid]['score']
            for k, v in discard_bboxes.items():
                iou = get_iou(bbox, v, method='col,row,square')
                score = min(score, 1.0 - iou)
            persons[pid]['score'] = score
    print('')
 
def debug_mode(args):

    if args.video:
        images = prepare_images(args)
    else:
        images = load_yaml(args.yaml_dir)

    if  args.dump_person:
        detection_graph = object_detection.prepare_model()
        images = object_detection.detect(images, detection_graph)
        ppaths = save_person(images, args.person_dir)
        save_yaml(images, args.yaml_dir)
        if ppaths:
            emb = person_reid.embed(ppaths)
            person_reid.update_images_with_emb(images, ppaths, emb)
            save_yaml(images, args.yaml_dir)

    if args.dump_face:
        # pnet, rnet, onet = face_detection.prepare_model()
        # images = face_detection.detect(images, pnet, rnet, onet)
        pass

    if args.discard_yml:
        discard_bbox(images, args.discard_yml)
        save_yaml(images, args.yaml_dir)

    if args.name_dir:
        name_vectors = embed_name_vector(args.name_dir)
        distance = person_reid.get_distance(name_vectors)
        print('distance is {}'.format(distance))
        person_reid.reid(images, name_vectors, distance)
        save_yaml(images, args.yaml_dir)
        
    if args.dump_video:
        image_utils.save_video_from_image(args.output_dir, images)

def display_mode(args):
    for image in image_utils.read_image_from_video(args.video):
        continue

def main():
    args = parse_args()

    if args.debug:
        debug_mode(args)
    else:
        if not args.video:
            print('--video is required for non debug mode')
            exit(1)
        if not args.name_dir:
            print('--name_dir is required for non debug mode')
        display_mode(args)

if __name__ == '__main__':
    main()
