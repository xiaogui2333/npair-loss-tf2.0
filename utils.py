#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_utils.py

import cv2
import numpy as np
import argparse
import random
import math
import sys
import os

from tensorpack.dataflow import (
    BatchData, MultiThreadMapData, DataFromList)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
import tensorflow as tf
import multiprocessing

class ILSVRC12Files(RNGDataFlow):
    """
    Same as :class:`ILSVRC12`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """
    def __init__(self, dir, shuffle=None):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        self.imglist = dir

    def __len__(self):
        return len(self.imglist)

    def __iter__(self):
        idxs = np.arange(len(self.imglist))
        for k in idxs:
            fname, label = self.imglist[k]
            yield [fname, label]

def ReadImageToCVMat(filename, args):
    height = args.height
    width = args.width
    is_color = args.is_color
    # max_aspect_ratio = args.max_aspect_ratio
    # min_aspect_ratio = args.min_aspect_ratio
    cv_img = None

    cv_read_flag = cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE
    cv_img_origin = cv2.imread(filename, cv_read_flag)

    orig_width = cv_img_origin.shape[1]
    orig_height = cv_img_origin.shape[0]
    aspect_ratio = orig_width / orig_height
    if height > 0 and width > 0:
    	target_as = width / height
    	cv_img = np.zeros([height, width, cv_img_origin.shape[2]], dtype=np.uint8)
    	if target_as < aspect_ratio:
			# which means target is slimmer
    		scale = width / orig_width
    		scaled_width = width
    		scaled_height = min(height, int(scale * orig_height))
    		target_img = cv2.resize(cv_img_origin, (scaled_width, scaled_height))
    		start_x = 0
    		start_y = 0
    		cv_img[start_y: start_y + scaled_height, start_x: start_x + scaled_width] = target_img
    	else:
    		scale = height / orig_height
    		scaled_width = min(width, int(scale * orig_width))
    		scaled_height = height
    		target_img = cv2.resize(cv_img_origin, (scaled_width, scaled_height))
    		start_x = int((width - scaled_width) / 2)
    		start_y = 0
    		cv_img[start_y: start_y + scaled_height, start_x: start_x + scaled_width] = target_img
    else:
    	cv_img = cv_img_origin
    return cv_img

def transform(cv_img, args):
	
    crop_size = args.crop_size
    mirror = args.mirror
    mean_values_ = args.mean_values_
    phase = args.phase
    scale = args.scale

    img_channels = cv_img.shape[2]
    img_height = cv_img.shape[0]
    img_width = cv_img.shape[1]

    do_mirror = mirror and random.randint(0, 1)

    assert img_channels > 0
    assert img_height >= crop_size
    assert img_width >= crop_size

    has_mean_values = len(mean_values_) > 0

    transformed_img = cv_img

    if crop_size:
    	if phase == "Train":
    		h_off = random.randint(0, img_height - crop_size + 1)
    		w_off = random.randint(0, img_width - crop_size + 1)
    	else:
    		h_off = (img_height - crop_size) // 2
    		w_off = (img_width - crop_size) // 2
    	transformed_img = cv_img[h_off: h_off + crop_size, w_off: w_off + crop_size]
    if do_mirror:
    	transformed_img = cv2.flip(transformed_img, 1)

    if has_mean_values:
        transformed_img = transformed_img - np.array(mean_values_).reshape([1,1,3])
    else:
    	transformed_img = transformed_img * scale
    transformed_img = transformed_img.astype(np.float32)
    return transformed_img

def occ_augImage(srcColor):
	height = srcColor.shape[0]
	width = srcColor.shape[1]

	prob = 0.5
	this_prob = random.uniform(0, 1)
	if this_prob > prob:
		return srcColor
	sl = 0.02
	sh = 0.2
	as_r = 0.3
	for i in range(100):
		area = height * width
		scale = random.uniform(sl, sh)
		target_area = scale * area
		aspect_ratio = random.uniform(as_r, 1/as_r)
		h = int(math.sqrt(target_area * aspect_ratio))
		w = int(math.sqrt(target_area / aspect_ratio))
		x1 = 0
		x2 = 0
		if w < width and h < height:
			x1 = random.randint(0, height - h)
			x2 = random.randint(0, width - w)
			srcColor[x1: x1 + h, x2: x2 + w] = np.random.randint(0, 255, (h, w, 3))
			break
	return srcColor

def get_train_dataflow(datadir, args):
    parallel = min(args.process_nums, multiprocessing.cpu_count() // 2)
    ds = ILSVRC12Files(datadir, shuffle=False)

    def mapf(dp):
        fname, cls = dp
        im_0 = ReadImageToCVMat(fname[0], args)
        im_1 = ReadImageToCVMat(fname[1], args)

        im_0 = occ_augImage(im_0)
        im_1 = occ_augImage(im_1)

        im_0 = transform(im_0, args)
        im_1 = transform(im_1, args)
        
        return np.array([im_0, im_1]), np.array([cls, cls])

    ds = MultiThreadMapData(ds, parallel, mapf,
                            buffer_size=min(2000, ds.size()), strict=True)
    # ds = MultiProcessRunnerZMQ(ds, parallel)
    # ds = BatchData(ds, batch_size, remainder=True)
    # do not fork() under MPI
    return ds
	
def get_test_dataflow(datadir, args):
    parallel = min(args.process_nums, multiprocessing.cpu_count())
    ds = ILSVRC12Files(datadir, shuffle=False)

    def mapf(dp):
        fname, cls = dp
        im_0 = ReadImageToCVMat(fname, args)
        im_0 = transform(im_0, args)
        
        return np.array(im_0), np.array(cls)

    ds = MultiThreadMapData(ds, parallel, mapf,
                            buffer_size=min(2000, ds.size()), strict=True)
    # ds = BatchData(ds, batch_size, remainder=True)
    # do not fork() under MPI
    return ds


# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("--height", default=384)
# 	parser.add_argument("--width", default=128)
# 	parser.add_argument("--channels", default=3)
# 	parser.add_argument("--is_color", default=True)

# 	args = parser.parse_args()
# 	img = ReadImageToCVMat("C:\\Users\\xukaiping\\Desktop\\ch02002_20181031103725_pano_00000621_00001800.jpg", args)
# 	cv2.imshow("src", img)
# 	cv2.waitKey(0)
# 	img = occ_augImage(img)
# 	cv2.imshow("out", img)
# 	cv2.waitKey(0)
