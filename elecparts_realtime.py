#!/bin/python

import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from subprocess import Popen, PIPE

from elecparts_config import elecpartsConfig

import model as modellib
from unreal_utils import MODEL_DIR, limit_GPU_usage, compute_mean_AP

from elecparts_dataset import elecpartsDataset
import visualize


class InferenceConfig(elecpartsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights("/nas/datashare/datasets/elecparts/3/results/epoch_009/mask_rcnn_elecparts_0009.h5", by_name=True)
dataset = elecpartsDataset()

cv2.namedWindow("detector")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

#plt.ion()
#fig = plt.figure()
#sub = fig.add_subplot(1,1,1)
p = Popen(['ffmpeg', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', '1', '-i', '-', '-f', 'matroska', '-'], stdin=PIPE, stdout=1)
while rval:
    #cv2.imshow("detector", frame)
    rval, frame = vc.read()
    results = model.detect([frame], verbose=0)
    r = results[0]

    for j in range(len(r["rois"])):
        if r["scores"][j] >= 0.85:
            label = dataset.my_class_names[r["class_ids"][j]]

    #sub.clear()
    img = visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], dataset.my_class_names, r['scores'], ax=plt.subplots(1,1,figsize=(8,8))[1], score_threshold = 0.85)

    pic = Image.fromarray(img)
    pic.save(p.stdin, "JPEG")
    print("saved a frame")
    #os.write(1, img)
    #plt.pause(0.05)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
p.stdin.close()
cv2.destroyWindow("detector")
