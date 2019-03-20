#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Face embedding Thrift server."""
import pickle
from keras.models import load_model
import cv2
from gen_py.face_emb import FaceEmbedding
import numpy as np
import config
import tensorflow as tf
from keras import backend as K
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket
from thrift.transport import TTransport


def preprocess_input(x, data_format=None, version=1):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 93.5940
            x[:, 1, :, :] -= 104.7624
            x[:, 2, :, :] -= 129.1863
        else:
            x = x[..., ::-1]
            x[..., 0] -= 93.5940
            x[..., 1] -= 104.7624
            x[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 91.4953
            x[:, 1, :, :] -= 103.8827
            x[:, 2, :, :] -= 131.0912
        else:
            x = x[..., ::-1]
            x[..., 0] -= 91.4953
            x[..., 1] -= 103.8827
            x[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x


class FaceEmbeddingHandler:
    """Request handler class."""
    def get_emb_numpy(self, numpy_imgs):
        images= []
        for pkl in numpy_imgs:
            img = pickle.loads(pkl)
            img = cv2.resize(img, (image_size, image_size))
            img = img.astype('float64')
            # img = img[:, :, ::-1]
            images.append(img)

        images = np.array(images)
        images = preprocess_input(images, version=2)
        with graph.as_default():
            with sess.as_default():
                embeddings = vgg_features.predict(images)
        embeddings.shape = (-1, 2048)
        return embeddings


def main():
    print('Loading model...')
    global sess
    global graph
    global vgg_features
    global image_size

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    with tf.Graph().as_default() as graph:
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default() as sess:
            # vgg_features = VGGFace(model='resnet50', include_top=False, pooling='avg')
            vgg_features = load_model('models/reduce_model.h5',compile=False)
            image_size = 224
            handler = FaceEmbeddingHandler()
            processor = FaceEmbedding.Processor(handler)
            transport = TSocket.TServerSocket(
                host='0.0.0.0', port=config.THRIFT_PORT)
            tfactory = TTransport.TBufferedTransportFactory()
            pfactory = TBinaryProtocol.TBinaryProtocolFactory()
            server = TServer.TThreadedServer(
                processor, transport, tfactory, pfactory)
            print('READY')
            try:
                server.serve()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
