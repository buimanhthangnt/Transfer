from face_detection.face_detection_mtcnn import detect_face
import os
import glob
import sys
import cv2


data_path = sys.argv[1]
for fp in glob.glob(os.path.join(data_path, '*/*')):
    image = cv2.imread(fp)
    bbs, _ = detect_face(image[:,:,::-1])
    if len(bbs) == 0:
        os.remove(fp)
    else:
        l,t,r,b = bbs[0]
        face = image[t:b,l:r]
        cv2.imwrite(fp, face)
