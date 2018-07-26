from importlib import import_module
from itertools import count
import os
import sys
import zipfile
import re

import numpy as np
import tensorflow as tf

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

def embed(ppaths):
    ppaths = np.array(ppaths)
    dataset = tf.data.Dataset.from_tensor_slices(ppaths)
    dataset = dataset.map(
        lambda ppath: fid_to_image(ppath), 
        num_parallel_calls=8)

    batch_size = 256
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    images = dataset.make_one_shot_iterator().get_next()

    prepare_model()
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
        return emb_storage.tolist()

def update_images_with_emb(image_dict, ppaths, emb_storage):
    print('')
    for i, emb in enumerate(emb_storage):
        ppath = ppaths[i]
        base_name = os.path.basename(ppath)
        m = re.search('(\d+)_person_(\d+)', base_name)
        fid = str(m.group(1)).zfill(6)
        pid = int(m.group(2))
        image_dict[fid]['detections']['person'][pid]['embedding'] = emb
        print('\rUpdating image dict for fid={}, pid={}'.format(fid, pid), flush=True, end='')
    print('')

def get_distance(name_vectors):
    distance = 0.0
    mean_vectors = []
    for k, v in name_vectors.items():
        vectors = v['vectors']
        mean_vectors.append(np.array(vectors).mean(axis=0))
    for i in range(len(mean_vectors) - 1):
        for j in range(i+1, len(mean_vectors)):
            cur_distance = np.linalg.norm(mean_vectors[i] - mean_vectors[j])
            if distance > 0.0:
                distance = min(cur_distance, distance)
            else:
                distance = cur_distance

    return distance / 1.42

def reid(image_dict, name_vectors, distance):
    for fid in sorted(image_dict.keys()):
        persons = image_dict[fid]['detections']['person']
        if not persons: continue
        for pid in sorted(persons.keys()):
            if 'distance' in persons[pid]: del persons[pid]['distance']
            if 'candidate' in persons[pid]: del persons[pid]['candidate']
            if 'color' in persons[pid]: del persons[pid]['color']
            if 'rscore' in persons[pid]: del persons[pid]['rscore']

            emb = persons[pid]['embedding']
            can_name = None
            can_distance = 0.0
            can_color = '#ffffff'
            for name, v in name_vectors.items():
                for vector in v['vectors']:
                    cur_distance = np.linalg.norm(np.array(emb) - np.array(vector))
                    if not can_name or can_distance > cur_distance:
                        can_name = name
                        can_distance = cur_distance
                        can_color = v['color']
            persons[pid]['distance'] = int(can_distance)
            persons[pid]['candidate'] = can_name
            persons[pid]['color'] = can_color
            dscore = 60
            if can_distance <= distance:
                persons[pid]['rscore'] = 100
            elif can_distance <=  distance * 3:
                a = dscore - 100
                b = 100 - a
                persons[pid]['rscore'] = int(a * can_distance / distance + b)
            else:
                persons[pid]['rscore'] = dscore

