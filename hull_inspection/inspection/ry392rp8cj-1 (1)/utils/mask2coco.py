#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask2coc: 

This file takes the labeled images and produces a binary mask. 
Subsequently, it transforms the binary mask into coco json format. 

"""

import numpy as np
import os
import cv2 as cv
import json
import pycocotools

__author__ = "K. Kamzelis, G. Chliveros, and I. Tzanetatos"
__copyright__ = "Copyright 2019, WatchOver Project"
__license__ = "LGPL"
__version__ = "1.0"
__maintainer__ = "Tzanetatos / Chliveros"
__email__ = "mech-tech@mitropolitiko.edu.gr"

def _loadData(path):
    
    imgs = []
    ids = []
    
    for img in os.listdir(path):
        if img.endswith(".jpg"):
            imgs.append(cv.imread(os.path.join(path, img)))
            
            ids.append(img)
    
    return imgs, ids

def _images(img, path, ids):
    
    
    img = { "id" : ids[:-3],
           "url" : "http://dx.doi.org/10.17632/ry392rp8cj.1",
           "width" : img.shape[1],
           "height" : img.shape[0],
           "file_name" : ids,
           "path" : os.path.join(path, ids)}
    
    return img

def _annotations(img, ids, anId, catId=1):
    
   
    # Ensure label is binary
    label = np.where(img[:,:,0] > 0, np.uint8(1), np.uint8(0))
    
    encMask = np.where(img[:, :, :] > 0, np.uint8(1), np.uint8(0))
    encMask = np.asfortranarray(encMask)
    
    area = img.shape[0] * img.shape[1] - np.sum(label)
    
    # counts = _mask2rle(encMask)
    rle = pycocotools.mask.encode(encMask)
    
    if type(rle) is list:
        for i in range(len(rle)):
            rle[i]['counts'] = rle[i]['counts'].decode("utf-8")
    else:
        rle['counts'] = rle['counts'].decode('ascii-8')
    
    annot = {"id" : anId,
             "category_id" : catId,
             "segmentation" : rle,
             "image_id": ids,
             "area" : area,
             "bbox" : [0, 0, img.shape[1]-1, img.shape[0]-1]}
    
    return annot

def _cocoAnnot(imgs, labels, path, ids):
    
    
    annotations = {"info":
                   {"description": "MaVeCoDD Dataset: Marine Vessel Hull Corrosion in Dry-Dock Images",
                   "url" : "http://dx.doi.org/10.17632/ry392rp8cj.1",
                   "version" : "1.0",
                   "year" : 2021,
                   "contributor" : "K. Kamzelis, G. Chliveros, and I. Tzanetatos"},
        "licenses" : {
            "url" : "https://creativecommons.org/licenses/by/4.0/",
            "name" : "Attribution 4.0 International"},
        "images" : list(),
        "annotations": list(),
        "categories": [
            {"name" : "corrosion",
             "id" : 1}]
        }
    
    for i in range(len(imgs)):
        
        annotations["images"].append(_images(imgs[i], path, ids[i]))
        
        annotations["annotations"].append(_annotations(labels[i], ids[i], anId=int(i+1e6)))
    
    return json.dumps([annotations])

def _main():
    
    rawHiRes = '../HiRes/raw/'
    labelHiRes = '../HiRes/labeled/'
    
    rawLoRes = '../LoRes/raw/'
    labelLoRes = '../LoRes/labeled/'
    
    HiRes, names = _loadData(rawHiRes)
    labHiRes, _ = _loadData(labelHiRes)
    
    HiResAnnot = _cocoAnnot(HiRes, labHiRes, rawHiRes, names)
    
    LoRes, names = _loadData(rawLoRes)
    labLoRes, _ = _loadData(labelLoRes)
    
    LoResAnnot = _cocoAnnot(LoRes, labLoRes, rawLoRes, names)
    
    with open('../HiRes/labeled/annotations.json', 'w') as file:
        json.dump(HiResAnnot, file)
    
    with open('../LoRes/labeled/annotations.json', 'w') as file:
        json.dump(LoResAnnot, file)

if __name__ == '__main__':
    _main()