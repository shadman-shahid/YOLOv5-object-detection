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


class DetectYolov5():
    def __init__(self, model_path):
        if torch.cuda.is_available():
            print('Running on CUDA Device')
            self.device = torch.device('cuda:0')
            print('Device Name : ',torch.cuda.get_device_name(0))
        else:
            print('Running on CPU')
            self.device = torch.device('cpu')
            
        self.model_path = model_path
        self.model = attempt_load(self.model_path, map_location = self.device) 
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names 


    def detect_image(self, image, conf_threshold = 0.25, iou_threshold = 0.45, resize_factor = None):
        img = LoadImages(image, self.device) 
        if resize_factor != None:
            image = cv2.resize(image, (0, 0), fx = resize_factor, fy = resize_factor)
        pred = self.model(img)[0]
        max_det = 1000
        classes = None
        agnostic_nms = False
        pred = non_max_suppression(pred, conf_threshold, iou_threshold, classes, agnostic_nms, max_det=max_det)
        source = " "
        path = source
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
        names = []
        bboxes = []

        im2 = image.copy()
        for i in range (len(det)):
            #left, top, right, bottom = np.array(det[i,0:4])
            left, top, right, bottom = det[i,0:4].cpu().detach().numpy()
            bbox = [left, top, right, bottom]
            bboxes.append(bbox)
            #index = int(np.array(det[i,5]))
            index = int(det[i,5].cpu().detach().numpy())
            #score = round(float(np.array(det[i,4])),2)
            score = round(float(np.array(det[i,4].cpu().detach().numpy())),2)
            names.append(self.names[index])
            cv2.rectangle(im2, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            #cv2.rectangle(im2, (int(left), int(top) + 15), (int(right), int(top)), (0, 255, 255), cv2.FILLED)
            cv2.rectangle(im2, (int(left), int(top) + 15), (int(left)+len(self.names[index])*13, int(top)), (0, 255, 255), cv2.FILLED)
            cv2.putText(im2, self.names[index], (int(left) + 2, int(top) + 12), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), thickness = 1, lineType = 2)
            
        return im2, bboxes, names

