from importlib import import_module
from itertools import count
import os
import sys
import zipfile
import re

import numpy as np
import tensorflow as tf

import image_utils

def prepare_model():
    base_dir = 'models/person_reid'
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    cp_path = os.path.join(base_dir, 'market1501_weights')
    link = 'https://github.com/VisualComputingInstitute/triplet-reid/releases/download/250eb1/market1501_weights.zip'
    zip_name = os.path.basename(link)
    zip_path = os.path.join(base_dir, zip_name)

    if not os.path.isfile(cp_path):
        if not os.path.isfile(zip_path):
            os.system('wget {}'.format(link))
            os.rename(zip_name, zip_path)
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(cp_path)
        zip_ref.close()

def fid_to_image(image_path):

    """ Loads and resizes an image given by FID. Pass-through the PID. """
    # Since there is no symbolic path.join, we just add a '/' to be sure.
    image_encoded = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)
    image_resized = tf.image.resize_images(image_decoded, (256,128))
    return image_resized

def embed(image_dict):
    prepare_model()

    ppaths = []
    for fid, fimage in image_dict.items():
        for pid, v in fimage['detections']['person'].items():
            ppaths.append(v['image_path'])
    ppaths = np.array(ppaths)

    dataset = tf.data.Dataset.from_tensor_slices(ppaths)
    dataset = dataset.map(
        lambda ppath: fid_to_image(ppath), 
        num_parallel_calls=8)

    batch_size = 256
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    images = dataset.make_one_shot_iterator().get_next()

    # Create the model and an embedding head.
    sys.path.append(os.path.abspath('models/person_reid'))
    model = import_module('nets.resnet_v1_50')
    head = import_module('heads.fc1024')

    endpoints, body_prefix = model.endpoints(images, is_training=False)
    embedding_dim = 128
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, embedding_dim, is_training=False)

    with tf.Session() as sess:
        checkpoint = 'models/person_reid/market1501_weights/checkpoint-25000'
        print('Restoring from checkpoint: {}'.format(checkpoint))
        tf.train.Saver().restore(sess, checkpoint)

        # Go ahead and embed the whole dataset, with all augmented versions too.
        emb_storage = np.zeros(
            (len(ppaths), embedding_dim), np.float32)
        for start_idx in count(step=batch_size):
            try:
                emb = sess.run(endpoints['emb'])
                print('\rEmbedded batch {}-{}/{}'.format(
                        start_idx, start_idx + len(emb), len(emb_storage)),
                    flush=True, end='')
                emb_storage[start_idx:start_idx + len(emb)] = emb
            except tf.errors.OutOfRangeError:
                break  # This just indicates the end of the dataset.

        print("Done with embedding, aggregating augmentations...", flush=True)
        print('')
        for i, emb in enumerate(emb_storage):
            print('\rUpdating image dict for fid={}, pid={}'.format(fid, pid), flush=True, end='')
            ppath = ppaths[i]
            base_name = os.path.basename(ppath)
            m = re.search('(\d+)_person_(\d+)', base_name)
            fid = str(m.group(1)).zfill(6)
            pid = int(m.group(2))
            image_dict[fid]['detections']['person'][pid]['embedding'] = emb.tolist()
        print('')

def get_distance(name_vectors):
    distance = 0.0
    for name1 in name_vectors.keys():
        for name2 in name_vectors.keys():
            if name1 == name2: continue
            for vector1 in name_vectors[name1]:
                for vector2 in name_vectors[name2]:
                    cur_distance = np.linalg.norm(np.array(vector1) - np.array(vector2))
                    if distance > 0.0:
                        distance = min(cur_distance, distance)
                    else:
                        distance = cur_distance
    return distance / 2


def reid(image_dict, name_vectors, distance):
    for fid in sorted(image_dict.keys()):
        persons = image_dict[fid]['detections']['person']
        if not persons: continue
        for pid in sorted(persons.keys()):
            if 'distance' in persons[pid]:
                del persons[pid]['distance']
                del persons[pid]['candidate']

            emb = persons[pid]['embedding']
            can_name = None
            can_distance = 0.0
            for name, vectors in name_vectors.items():
                for vector in vectors:
                    cur_distance = np.linalg.norm(np.array(emb) - np.array(vector))
                    if not can_name:
                        can_name = name
                        can_distance = cur_distance
                    elif can_distance > cur_distance:
                        can_name = name
                        can_distance = cur_distance
            persons[pid]['distance'] = int(can_distance)
            persons[pid]['candidate'] = can_name

