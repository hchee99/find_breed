#라벨만들기 위한 os, json
import os, json, re
import matplotlib.pyplot as plt 

#비전 관련 작업 cv로 진행할 때, numpy를 함께 임포트
import cv2 
import numpy as np

import shutil #.sh / .bash

#실제 코드 augmentation.py -> 호출 main.py
from Utils import augmentation as aug
import albumentations as A

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

if __name__ == '__main__':
    
    