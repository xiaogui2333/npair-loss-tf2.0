import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
import numpy as np
import random
import cv2
import math

def ReadImageToCVMat(filename, args):
    height = args.height
    width = args.width
    is_color = args.is_color
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
    mirror = args.mirror
    mean_values_ = args.mean_values_

    do_mirror = mirror and random.randint(0, 1)
    has_mean_values = len(mean_values_) > 0
    transformed_img = cv_img

    if do_mirror:
    	transformed_img = cv2.flip(transformed_img, 1)

    if has_mean_values:
    	transformed_img = transformed_img - np.array(mean_values_).reshape([1,1,3])

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
	
