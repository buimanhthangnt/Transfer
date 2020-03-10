import os
import glob
from keras.models import load_model
import imageio
import imutils
import cv2
import sys
sys.path.append('card_detection')
import numpy as np
import time
from sklearn.utils import shuffle
import tensorflow as tf
# from card_detection.card_detector import CardDetector


# detector = CardDetector()

with tf.Graph().as_default() as graph:
    with tf.InteractiveSession().as_default() as sess:
        model = load_model('models/passport_rotation_model.h5')
IDX2NAME = {
    0: 0,
    1: 180,
    2: 270,
    3: 90
}


def predict(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    with graph.as_default():
        with sess.as_default():
            pred = model.predict(image)[0]
    pred = np.argmax(pred)
    return pred


if __name__ == '__main__':
    times = []
    for idx, fp in enumerate(shuffle(glob.glob(os.path.join(sys.argv[1], '*')))):
        if idx % 100 == 0 and idx != 0:
            print("Done: ", idx)
            print(np.mean(np.array(times)))
        
        image = imageio.imread(fp)
        # image = imutils.resize(image, height=224)
        if image is None:
            continue
        start = time.time()
        with graph.as_default():
            with sess.as_default():
                pred = predict(image)
        times.append(time.time() - start)
        # print(IDX2NAME[pred])
        # print(IDX2NAME[pred])
        # cv2.imshow('test', image[:,:,::-1])
        # cv2.waitKey(0)
        if pred != 0:
            print(fp, IDX2NAME[pred])
            image = imutils.rotate_bound(image, IDX2NAME[pred])
            cv2.imwrite(fp, image[:,:,::-1])
