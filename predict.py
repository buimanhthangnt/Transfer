from imageio import imread
import os
import cv2
import client_thrift
import numpy as np
import csv
import sys
import glob
import pickle
import time
from scipy import spatial
import multiprocessing
import warnings
import config
import threading
from sklearn import svm
from numpy import dot
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from face_detection import detect_face
warnings.filterwarnings("ignore", category=UserWarning)


gs = pickle.load(open('models/gs.pkl', 'rb'))
svc = pickle.load(open(config.CLF_MODEL, 'rb'))
train_embs = pickle.load(open(config.TRAIN_EMB_PATH, 'rb'))
counts = pickle.load(open(config.COUNT_PATH, 'rb'))

emb_dict = {}
for key in train_embs:
    emb, label = train_embs[key]
    if label not in emb_dict:
        emb_dict[label] = []
    emb_dict[label].append(emb)
print("Load model done")

test_path = 'data/test'
all_data = glob.glob(os.path.join(test_path, '*'))
thread_data = []
num_data_each_thr = int(len(all_data) * 1.0 / config.NUM_THREADS)
for i in range(config.NUM_THREADS):
    if i < config.NUM_THREADS - 1:
        thread_data.append(all_data[i * num_data_each_thr:(i+1) * num_data_each_thr])
    else:
        thread_data.append(all_data[i * num_data_each_thr:])
# print([len(x) for x in thread_data])
# exit(0)


def match_prob(emb1, all_embs):
    X = []
    emb1 = np.array(emb1)
    for emb2 in all_embs:
        emb2 = np.array(emb2)
        euc = np.linalg.norm(emb1 - emb2)
        cos = 1 - spatial.distance.cosine(emb1, emb2)
        X.append([euc, cos])
    pred = gs.predict_proba(X)
    return list(pred[:,1])


def predict(target_emb):
    all_embs = [train_embs[x][0] for x in train_embs]
    all_labels = [train_embs[x][1] for x in train_embs]
    distances = list(zip(match_prob(target_emb, all_embs), all_labels))

    # distances = [(match_prob(target_emb, train_embs[x][0]), train_embs[x][1]) for x in train_embs]
    distances.sort(key=lambda x: x[0])
    distances.reverse()
    filtered_distances = distances[:config.K_KNN]
    count_match = [1 if x >= config.MATCH_THRESHOLD else 0 for (x, _) in filtered_distances]
    if sum(count_match) == 0 or (sum(count_match) == 1 and counts[filtered_distances[0][1]] >= 3):
        ret = [config.NUM_CLASSES]
        idx = 0
        while len(ret) < config.MAP:
            label = distances[idx][1]
            if label not in ret: ret.append(label)
            idx += 1
        return ret
    else:
        pred_proba = svc.predict_proba([target_emb])[0]
        pred_proba = [(idx, prob) for idx, prob in enumerate(pred_proba)]
        pred_proba.sort(key=lambda x: x[1])
        pred_proba.reverse()
        pred_proba = pred_proba[:config.MAP]
        pred_proba = [x[0] for x in pred_proba]
        pred_proba.insert(2, 1000)
        return pred_proba[:5]


def avg_similar(emb1, all_embs):
    X = []
    emb1 = np.array(emb1)
    for emb2 in all_embs:
        emb2 = np.array(emb2)
        cos = abs(1 - spatial.distance.cosine(emb1, emb2))
        X.append(cos)
    return np.mean(X)


def predict_2(target_emb):
    distances = []    
    for label in emb_dict:
        avg = avg_similar(target_emb, emb_dict[label])
        distances.append((avg, label))
    distances.sort(key=lambda x: x[0])
    distances.reverse()
    if distances[0][0] < config.COSINE_THRESHOLD:
        ret = [config.NUM_CLASSES]
        idx = 0
        while len(ret) < config.MAP:
            label = distances[idx][1]
            if label not in ret: ret.append(label)
            idx += 1
        return ret
    else:
        pred_proba = svc.predict_proba([target_emb])[0]
        pred_proba = [(idx, prob) for idx, prob in enumerate(pred_proba)]
        pred_proba.sort(key=lambda x: x[1])
        pred_proba.reverse()
        pred_proba = pred_proba[:config.MAP]
        pred_proba = [x[0] for x in pred_proba]
        pred_proba.insert(2, 1000)
        return pred_proba[:5]
    

def train_clf():
    train_embs = pickle.load(open(config.TRAIN_EMB_PATH, 'rb'))
    X, y = [], []
    for key in train_embs:
        emb, label = train_embs[key]
        X.append(emb)
        y.append(label)
    svc = LogisticRegression(random_state=23, solver='lbfgs', verbose=1, \
                             class_weight='balanced', multi_class='multinomial')
    # svc = svm.SVC(probability=True, kernel='linear', class_weight='balanced')
    svc.fit(X, y)
    pickle.dump(svc, open(config.CLF_MODEL, 'wb'), pickle.HIGHEST_PROTOCOL)


def main(filepaths):
    client, transport = client_thrift.create_socket()
    tmp_result = []
    for idx, fp in enumerate(filepaths):
        if idx % 5 == 0:
            print("Done: ", idx)
        # fp = os.path.join(test_path, fn)
        fn = fp.split('_')[-1]
        face = imread(fp)
        bbs, _ = detect_face(face)
        if len(bbs) > 0:
            l,t,r,b = bbs[0]
            face = face[t:b,l:r]
        emb = client_thrift.get_emb_numpy([face], client, transport)[0]
        labels = predict(emb)
        tmp_result.append([fn, ' '.join([str(x) for x in labels])])
        print(labels)
        for lb in labels[:4]:
            if lb == 1000: continue
            image_path = os.path.join("data/train_data", str(lb))
            image = imread(os.path.join(image_path, os.listdir(image_path)[0]))
            cv2.imshow('test' + str(lb), image[:,:,::-1])
        cv2.imshow('test', face[:,:,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return tmp_result


if __name__ == "__main__":
    idx = int(sys.argv[1])
    results = main(thread_data[idx])
    pickle.dump(results, open("res" + str(idx) + ".pkl", "wb"), pickle.HIGHEST_PROTOCOL)
    # for i in range(config.NUM_THREADS):
    #     thr = threading.Thread(target=main, kwargs={'filepaths': thread_data[i]})
    #     thr.setDaemon(True)
    #     thr.start()

    # main_thread = threading.current_thread()
    # for thr in threading.enumerate():
    #     if thr is main_thread: continue
    #     thr.join()

    # csv_content = [["image", "label"]]
    # csv_content.extend(results)
    # with open('output.csv', 'w') as fn:
    #     csv.writer(fn).writerows(csv_content)
