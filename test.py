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
    with tf.Session().as_default() as sess:
        model = load_model('models/ma_model_new.h5')
IDX2NAME = {
    0: 'no',
    1: 'yes'
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
    for idx, fp in enumerate(shuffle(glob.glob(os.path.join(sys.argv[1], '*/*')))):
        if idx % 100 == 0 and idx != 0:
            print("Done: ", idx)
            #print(np.mean(np.array(times)))
        ext = fp.split('.')[-1]
        label = fp.split('/')[-2]
        if ext.lower() not in ['jpg', 'jpeg', 'png']: continue
        image = imageio.imread(fp)
        image = imutils.resize(image, height=224)
        if image.shape[-1] != 3:
            continue
        # image, _ = detector.detect_card(image)
        if image is None:
            continue
        start = time.time()
        with graph.as_default():
            with sess.as_default():
                pred = predict(image)
        times.append(time.time() - start)
        # print(IDX2NAME[pred])
        if label != IDX2NAME[pred]:
            print(IDX2NAME[pred])
            cv2.imshow('test', image[:,:,::-1])
            cv2.waitKey(0)
            print(fp, '\n')
