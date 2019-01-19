# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:44:46 2018

@author: wmy
"""

import os
import random
import xml.etree.ElementTree as ET
from os import getcwd

anchors = [(10, 13), 
           (16, 30), 
           (33, 23), 
           (30, 61), 
           (62, 45), 
           (59, 119), 
           (116, 90), 
           (156, 198), 
           (373, 326)]

classes = ['anime face']

image_extension = 'jpg'

test_percent = 0.1
val_percent = 0.1
train_precent = 0.8

def set_anchors():
    with open(r'./infos/anchors.txt', 'w') as file:
        for i in range(len(anchors)):
            if i != len(anchors)-1:
                info = str(anchors[i][0]) + ',' + str(anchors[i][1]) + \
                ',' + ' '*2
            else:
                info = str(anchors[i][0]) + ',' + str(anchors[i][1])
            file.write(info)
            pass
        pass
    print('[ok] anchors setted')
    pass

def set_classes():
    with open(r'./infos/classes.txt', 'w') as file:
        for class_name in classes:
            file.write(class_name + '\n')
            pass
        pass
    print('[ok] classes setted')
    pass

def convert_annotation(image_id, list_file):
    in_file = open('dataset/annotations/%s.xml' % image_id)
    tree=ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        pass
    pass

def split_dataset():
    xml_path = r'./dataset/annotations'
    xmls = os.listdir(xml_path)
    num = len(xmls)
    n_test = int(num * test_percent)
    n_val = int(num * val_percent)
    n_train = num - n_test - n_val
    train_val = random.sample(xmls, n_train+n_val)
    train = random.sample(train_val, n_train)
    f_train = open(r'./infos/train.txt', 'w')
    f_val = open(r'./infos/val.txt', 'w')
    f_test = open(r'./infos/test.txt', 'w')
    for xml in xmls:
        if xml in train_val:
            if xml in train:
                name = xml[:-4]
                f_train.write('dataset/images/%s.%s' % (name, image_extension))
                convert_annotation(name, f_train)
                f_train.write('\n')
                pass
            else:
                name = xml[:-4]
                f_val.write('dataset/images/%s.%s' % (name, image_extension))
                convert_annotation(name, f_val)
                f_val.write('\n')
                pass
            pass
        else:
            name = xml[:-4]
            f_test.write('dataset/images/%s.%s' % (name, image_extension))
            convert_annotation(name, f_test)
            f_test.write('\n')
            pass
        pass
    f_train.close()
    f_val.close()
    f_test.close()
    print('[ok] dataset splited')
    pass

if __name__ == '__main__':
    set_anchors()
    set_classes()
    split_dataset()
