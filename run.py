# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 12:30:21 2018

@author: wmy
"""

import sys
import argparse
from PIL import Image
import glob
import os
from predict import YOLO, detect_video, predict_testset, predict_valset, predict_trainset

def run(yolo, path, outdir):     
    for jpgfile in glob.glob(path):
        img = Image.open(jpgfile)
        img = yolo.detect_image(img)
        img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        pass
    pass

if __name__ == '__main__':
    yolo = YOLO()
    predict_trainset(yolo)
    predict_valset(yolo)
    predict_testset(yolo)