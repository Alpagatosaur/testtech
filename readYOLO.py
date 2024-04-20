# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:53:15 2024

@author: tanch

READ LP
"""

import cv2
import os
from PIL import Image
import torch
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
from glob import glob

from detect import main as runmod

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

list_anb = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def parse():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    
    """Config model yolov5"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/exp6/weights/best.pt", help="model path or triton URL")
    parser.add_argument("--data", type=str, default=ROOT / "datasets/data.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.40, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_false", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_false", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "temp", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    return parser

def get_info():
    parser = parse()
    img = glob("temp\*.jpg")[-1]
    parser.add_argument("--source", type=str, default=img, help="file/dir/URL/glob/screen/0(webcam)")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    runmod(opt)

def sorted_from(l, pos=0):
    
    new_list, copy_list = [], l
    while len(l) != 0:
        mini = l[0]
        for line in l:
            if line[pos] < mini[pos]:
                mini = line
        new_list.append(mini)
        copy_list.remove(mini)
    return new_list
            
        
def read_lp():
    get_info()
    lines = []
    if os.path.isdir('temp/exp'):
        file = glob("temp/exp/labels/*.txt")[0]
        with open(file) as f:
            for line in f:
                ttemp = []
                for elt in line.rstrip().split(" "):
                    ttemp.append(float(elt))
                lines.append(ttemp)

        lines = sorted_from(lines, pos=1)
        text = ""
        for line in lines:
            ind = int(line[0])
            text = text + str(list_anb[ind]) + " "
        return text

if __name__ == "__main__":
    t = read_lp()
    print(t)
                
        
        