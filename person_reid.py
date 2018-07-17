from importlib import import_module
from itertools import count
import os
import sys
import zipfile

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
        opener = urllib.request.URLopener()
        opener.retrieve(link, zip_path)
        zip_ref = zipfile.ZipFile(zip_path, 'r')
        zip_ref.extractall(cp_path)
        zip_ref.close()

def prepare_dataset(images):
    image_nps, pids, fids = [], [], []
    for fid, fimage in images.items():
        for pid, v in fimage['detections']['person'].items():
            image_np = image_utils.read_image_to_np(v['image_path'])
            image_nps.append(image_np)
            pids.append(pid)
            fids.append(fid)
    return tf.data.Dataset.from_tensor_slices(image_nps) pids, fids

def fid_to_image(fid, pid, image_root):
    """ Loads and resizes an image given by FID. Pass-through the PID. """
    image_encoded = tf.read_file(tf.reduce_join([image_root, '/', fid]))
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=3)

    return image_decoded, fid, pid

def embed(images):
    prepare_model()

    dataset = prepare_dataset(images)
    batch_size = 256
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    images, pids, fids = dataset.make_one_shot_iterator().get_next()

    # Create the model and an embedding head.
    sys.path.append(os.path.abspath('models/person_reid'))
    model = import_module('nets.resnet_v1_50')
    head = import_module('heads.fc1024')

    endpoints, body_prefix = model.endpoints(images, is_training=False)
    embedding_dim = 128
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, embedding_dim, is_training=False)

    with tf.Session() as sess:
        checkpoint = 'models/person_reid/market1501_weights'
        print('Restoring from checkpoint: {}'.format(checkpoint))
        tf.train.Saver().restore(sess, checkpoint)

        # Go ahead and embed the whole dataset, with all augmented versions too.
        emb_storage = []
        for start_idx in count(step=batch_size):
            try:
                emb = sess.run(endpoints['emb'])
                print('\rEmbedded batch {}-{}/{}'.format(
                        start_idx, start_idx + len(emb), len(emb_storage)),
                    flush=True, end='')
                emb_storage.append(emb)
            except tf.errors.OutOfRangeError:
                break  # This just indicates the end of the dataset.

        print("Done with embedding, aggregating augmentations...", flush=True)
        result = zip(np.array(emb_storage).flatten(), pids, fids)
        print(result.shape)



if __name__ == '__main__':
    embed()
