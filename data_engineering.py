import glob
import os
import sys
import cv2
import imutils
import shutil
import imageio
import time
# from image_rotation.image_rotator import image_rotator
from sklearn.utils import shuffle
import random


def gen_float_number(xmin=0.05, xmax=0.25):
    return xmin + random.random() * (xmax - xmin)


def gen_miss_angle(image):
    height, width = image.shape[:2]
    x = gen_float_number()
    y = gen_float_number()
    #print("X, y", x, y)
    x_offset, y_offset = int(x * width), int(y * height)
    rand1 = random.random()
    if rand1 > 0.92:
        image = image[y_offset:height-y_offset, x_offset:width-x_offset]
    elif 0.7 <= rand1 <= 0.92:
        if random.random() > 0.5:
            image = image[:,x_offset:]
        else:
            image = image[:,:width - x_offset]

        if random.random() > 0.5:
            image = image[y_offset:,:]
        else:
            image = image[:height-y_offset,:]
    else:
        rand2 = random.random()
        if rand2 < 0.25:
            image = image[:,x_offset:]
        elif 0.25 <= rand2 < 0.5:
            image = image[:,:width - x_offset]
        elif 0.5 <= rand2 < 0.75:
            image = image[y_offset:,:]
        else:
            image = image[:height-y_offset,:]
    return image


def gen_data_miss_angle(inpath, outpath, num):
    miss_dir = os.path.join(outpath, "yes")
    normal_dir = os.path.join(outpath, "no")
    if not os.path.exists(miss_dir):
        os.makedirs(miss_dir)
    if not os.path.exists(normal_dir):
        os.makedirs(normal_dir)
    for idx, fp in enumerate(shuffle(glob.glob(os.path.join(inpath, 'cmt_*/*')))[:num]):
        if idx % 200 == 0:
            print("Done: ", idx)
        image = cv2.imread(fp)
        image = image_rotator.predict(image[:,:,::-1])[:,:,::-1]
        image = imutils.resize(image, height=224)
        cv2.imwrite(os.path.join(outpath, "no", os.path.split(fp)[1]), image)
        image = gen_miss_angle(image)
        cv2.imwrite(os.path.join(outpath, "yes", os.path.split(fp)[1]), image)


def test_split():
    train_dir = 'data/train'
    test_dir = 'data/test'
    test_ratio = 0.1
    if os.path.exists(test_dir):
        return
    for subset in os.listdir(train_dir):
        sub_train_dir = os.path.join(train_dir, subset)
        sub_test_dir = os.path.join(test_dir, subset)
        if not os.path.exists(sub_test_dir):
            os.makedirs(sub_test_dir)
        train_files = shuffle(glob.glob(os.path.join(sub_train_dir, '*')))
        test_size = int(len(train_files) * test_ratio)
        test_files = train_files[:test_size]
        for test_file in test_files:
            shutil.move(test_file, sub_test_dir)


def clean_noise():
    IDX2NAME = {
        0: 'cmt_back',
        1: 'cmt_front',
        2: 'others',
        3: 'passport_back',
        4: 'passport_front',
        5: 'si_quan_back',
        6: 'si_quan_front'
    }
    path = sys.argv[1]
    noise_path = sys.argv[2]
    for idx, fp in enumerate(shuffle(glob.glob(os.path.join(path, '*')))):
        if idx % 100 == 0 and idx != 0:
            print("Done: ", idx)
        ext = fp.split('.')[-1]
        label = "cmt_back"
        if ext.lower() not in ['jpg', 'jpeg', 'png']: continue
        image = imageio.imread(fp)
        # image = imutils.resize(image, height=224)
        if image.shape[-1] != 3:
            continue
        pred = predict(image)
        # print(IDX2NAME[pred])
        if label != IDX2NAME[pred]:
            print(IDX2NAME[pred], label)
            # cv2.imshow('test', image[:,:,::-1])
            # cv2.waitKey(0)
            # outpath = os.path.join(noise_path, label)
            # if not os.path.exists(outpath):
            #     os.makedirs(outpath)
            # shutil.copyfile(fp, os.path.join(outpath, fp.split('/')[-1]))
            shutil.move(fp, noise_path)
            # os.remove(fp)
            print("Moved ", fp)
            # time.sleep(2)


def try_image():
    import cv2
    image = cv2.imread(sys.argv[1])
    image = gen_miss_angle(image)
    cv2.imshow('test', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    # clean_noise()
    # gen_data_miss_angle("/home/common_gpu0/corpora/vision/idreader/id_types/train", "data/train", 20000)
    test_split()
    pass        
