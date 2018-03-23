#!/bin/python

import os
import cv2
from subprocess import Popen, PIPE

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import model as modellib
from unreal_utils import MODEL_DIR, limit_GPU_usage, compute_mean_AP
from elecparts_config import elecpartsConfig
from elecparts_dataset import elecpartsDataset
import visualize


class InferenceConfig(elecpartsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(("/nas/datashare/datasets/elecparts/5/results/elecparts2018"
        + "0318T0833/mask_rcnn_elecparts_0015.h5"), by_name=True)
dataset = elecpartsDataset()

cv2.namedWindow("detector")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()

    # Invoke the model to perform detection
    results = model.detect([frame], verbose=0)
    r = results[0]

    # Apply the colored masks
    img = visualize.display_instances(frame, r['rois'], r['masks'],
            r['class_ids'], dataset.my_class_names, r['scores'],
            ax=plt.subplots(1,1,figsize=(8,8))[1], score_threshold = 0.85)

    for j in range(len(r["rois"])):
        if r["scores"][j] >= 0.85:
            # Draw the bounding box
            y1, x1, y2, x2 = r["rois"][j]
            cv2.rectangle(img, (x1,y1), (x2,y2), (100,100,100), 2)

            # Draw the label
            label = dataset.my_class_names[r["class_ids"][j]]
            cv2.putText(img, label, (x1,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3/3, (255,255,255), 1, cv2.LINE_AA)

    # show the annotated frame next to the original
    final = np.concatenate((frame,img), axis=1)
    cv2.imshow("detector", final)
    
    key = cv2.waitKey(20)
    # exit on ESC
    if key == 27:
        break

vc.release()
cv2.destroyWindow("detector")
