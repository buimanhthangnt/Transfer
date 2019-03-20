import os
import pandas
import shutil
import glob
import client_thrift
import imageio
import cv2
import pickle
import config
from face_detection import detect_face


path = 'data/fr/train'
outpath = 'data/train_data'


def organize():
    data = pandas.read_csv('data/fr/train.csv', sep=',', header=None).values[1:]

    for sample in data:
        fn, label = sample
        outdir = os.path.join(outpath, label)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        shutil.move(os.path.join(path, fn), outdir)


def compute_train_emb():
    client, transport = client_thrift.create_socket()
    embs = {}
    for idx, fp in enumerate(glob.glob(os.path.join(outpath, '*/*'))):
        if idx % 100 == 0:
            print("Done: ", idx)
        fn = fp.split(os.sep)[-1]
        label = int(fp.split(os.sep)[-2])
        image = imageio.imread(fp)
        bbs, _ = detect_face(image)
        if len(bbs) > 0:
            l,t,r,b = bbs[0]
            image = image[t:b,l:r]
        feature = client_thrift.get_emb_numpy([image], client, transport)[0]
        embs[fn] = [feature, label]
    pickle.dump(embs, open(config.TRAIN_EMB_PATH, 'wb'), pickle.HIGHEST_PROTOCOL)


def count_n_sample():
    count = {}
    for subdir in os.listdir(outpath):
        count[int(subdir)] = len(os.listdir(os.path.join(outpath, subdir)))
    pickle.dump(count, open(config.COUNT_PATH, 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # organize()
    # compute_train_emb()
    # count_n_sample()
