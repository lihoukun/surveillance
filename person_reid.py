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

def fid_to_image(fid, pid, image_root):
    """ Loads and resizes an image given by FID. Pass-through the PID. """
    image_encoded = tf.read_file(tf.reduce_join([image_root, '/', fid]))
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)

    return image_decoded, fid, pid

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
        for i, emb in enumerate(emb_storage):
            ppath = ppaths[i]
            base_name = os.path.basename(ppath)
            m = re.search('(\d+)_person_(\d+)', base_name)
            fid = str(m.group(1)).zfill(6)
            pid = int(m.group(2))
            print(fid, pid)
            print('\rUpdaing image dict for fid={}, pid={}'.format(fid, pid), flush=True, end='')
            image_dict[fid]['detections']['person'][pid]['embedding'] = emb.tolist()


if __name__ == '__main__':
    embed()
