import os
import numpy as np
import pickle
import config
from scipy import spatial


path = 'data/train_data'
data = []
train_embs = pickle.load(open(config.TRAIN_EMB_PATH, "rb"))


for subdir in os.listdir(path):
    tmp = []
    for fn in os.listdir(os.path.join(path, subdir)):
        tmp.append(train_embs[fn][0])
    data.append(tmp)


def cosine(emb1, emb2):
    return 1 - spatial.distance.cosine(emb1, emb2)


same = []
for i in range(5000):
    x = np.random.randint(len(data))
    while len(data[x]) == 1:
        x = np.random.randint(len(data))
    m, n = np.random.randint(0, len(data[x]), size=2)
    same.append(cosine(data[x][m], data[x][n]))

diff = []
for i in range(5000):
    x, y = np.random.randint(0, len(data), size=2)
    m = np.random.randint(len(data[x]))
    n = np.random.randint(len(data[y]))
    diff.append(cosine(data[x][m], data[y][n]))

print(np.mean(same))
print(np.std(same))
print(np.mean(diff))
print(np.std(diff))
