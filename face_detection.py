from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import align.detect_face as mtcnn
import numpy as np


gpu_memory_fraction = 0.2
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = mtcnn.create_mtcnn(sess, './align')

threshold = [0.56, 0.75, 0.9]  # three steps's threshold
factor = 0.55  # scale factor


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def detect_face(img, min_size=40, max_size=800):
    if img.ndim < 2:
        print('Unable to detect face: img.ndim < 2')
        return None
    if img.ndim == 2:
        img = to_rgb(img)
    img = img[:, :, 0:3]
    bounding_boxes, points = mtcnn.detect_face(img, min_size, pnet, rnet, onet, threshold, factor)
    bbs, landmarks = [], []
    for idx, face_position in enumerate(bounding_boxes):
        face_position = face_position.astype(int)
        l, t, r, b = face_position[:4]
        if b - t < 2 or r - l < 2 or b < 0 or l < 0 or t < 0 or r < 0: continue
        bbs.append((l, t, r, b))
        landmarks.append(points[:,idx])
    landmarks = np.transpose(np.array(landmarks))
    return bbs, landmarks
