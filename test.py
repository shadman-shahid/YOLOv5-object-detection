import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.custom import LoadImages,non_max_suppression, scale_coords


device='cpu'
w='./model_data/yolov5s.pt'
model=attempt_load(w, map_location=device)

stride = int(model.stride.max())  
names = model.module.names if hasattr(model, 'module') else model.names  

source='random.jpeg'
image=cv2.imread(source)
img = LoadImages(image) 
pred = model(img)[0]

imgsz=640
conf_thres=0.25 
iou_thres=0.45  
max_det=1000
classes=None
agnostic_nms=False

pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

source=" "
path=source
det=pred[0]
det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

im2=image
for i in range (len(det)):
    left, top, right, bottom=np.array(det[i,0:4])
    index=int(np.array(det[i,5]))
    score=round(float(np.array(det[i,4])),2)
    print(names[index],score)
    cv2.rectangle(im2, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    cv2.rectangle(im2, (int(left), int(top) + 35), (int(right), int(top)), (0, 255, 0), cv2.FILLED)
    cv2.putText(im2, names[index], (int(left) + 20, int(top) + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)
    

small_frame = cv2.resize(im2, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('Frame',small_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()