import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import numpy as np
import random
import cv2
import os
import pdb
import time
import argparse
from data_transformer import ReadImageToCVMat, transform, occ_augImage

class  NpairDataClass(object):
	def __init__(self, args):
		self.root_folder = args.root_folder
		self.source = args.source

		self.new_height = args.height
		self.new_width = args.width
		self.is_color = args.is_color
		self.crop_size = args.crop_size
		self.mean_values_ = args.mean_values_
		self.args = args

		assert self.new_height > 0 and self.new_width > 0, "Current implementation requires "\
		"new_height and new_width to be set at the same time."

		self.class_vec_ = set()
		self.lines_ = []
		self.sec_labels_ = []
		self.seclabel_labels_map_ = {}
		self.class_lines_map_ = {}
		self.epoch = -1

		self.rand_skip = 0
		self.shuffle = args.shuffle

		self.occ_aug = args.occ_aug
		self.batch_size = args.batch
		self.phase = args.phase
		self.line = []
	def setUp(self):
		print("Opening file ", self.source)
		infile = open(self.source)
		sec_labels_set = set()

		for line in infile:
			line = line.strip().split(' ')
			filename = line[0]
			label = int(line[1])
			sec_label = int(line[2])
			sec_labels_set.add(sec_label)

			if sec_label not in self.seclabel_labels_map_.keys():
				self.seclabel_labels_map_[sec_label] = set()
			if sec_label not in self.class_lines_map_.keys():
				self.class_lines_map_[sec_label] = {}
			if label not in self.class_lines_map_[sec_label].keys():
				self.class_lines_map_[sec_label][label] = []

			self.seclabel_labels_map_[sec_label].add(label)
			self.lines_.append([filename, label])
			self.class_lines_map_[sec_label][label].append(len(self.lines_) - 1)

			self.line.append(line)
		self.sec_labels_ = list(sec_labels_set)

		for iteration in self.class_lines_map_.keys():
			self.class_vec_.update(list(self.class_lines_map_[iteration].keys()))

		print("A total of ", len(self.lines_), " images.")
		print("A total of ", len(self.class_vec_), " classes.")
		print("A total of ", len(self.sec_labels_), " second labels.")


	def ShuffleLeafsWithSecLabels(self, sec_label_num):
		self.seclabel_labels_map_[sec_label_num] = list(self.seclabel_labels_map_[sec_label_num])
		random.shuffle(self.seclabel_labels_map_[sec_label_num])
		return self.seclabel_labels_map_[sec_label_num]

	def next_iteration(self):
		files = []
		labels = []
		# print("num_class", num_classes)
		
		for _ in range(self.args.iterations_per_epoch):
			sec_label_num = np.random.choice(self.sec_labels_, 1)[0]
			leaf_labels = self.ShuffleLeafsWithSecLabels(sec_label_num)
			num_classes = len(leaf_labels)
			count = 0
			for item_id  in range(num_classes):
				class_label = leaf_labels[item_id]
				lines = self.class_lines_map_[sec_label_num][class_label]
				if len(lines)<2:
					continue
				idx = np.random.choice(lines, 2, replace=False)
				line_id_1 = idx[0]
				line_id_2 = idx[1]
				# files.append([self.root_folder+self.lines_[line_id_1][0], self.root_folder+self.lines_[line_id_2][0]])
				files.append(self.root_folder+self.lines_[line_id_1][0])
				files.append(self.root_folder+self.lines_[line_id_2][0])
				labels.append(self.lines_[line_id_1][1])
				labels.append(self.lines_[line_id_2][1])
				count += 1
				if count==self.batch_size:
					break
					
		return files, labels


	def prefetchData(self):
		sec_label_num = np.random.choice(self.sec_labels_, 1)[0]
		leaf_labels = self.ShuffleLeafsWithSecLabels(sec_label_num)
		# num_classes = len(leaf_labels)

		prefetch_data = []
		prefetch_label = []
		# print("num_classes", num_classes)
		for item_id  in range(len(leaf_labels)):
  			# get a batch data
			class_label = leaf_labels[item_id]
			lines = self.class_lines_map_[sec_label_num][class_label]
			if len(lines)<2:
				continue
			idx = np.random.choice(len(lines), 2, replace=False)
			line_id_1 = lines[idx[0]]
			line_id_2 = lines[idx[1]]
			cv_img_1 = ReadImageToCVMat(self.root_folder+self.lines_[line_id_1][0], self.args)
			cv_img_2 = ReadImageToCVMat(self.root_folder+self.lines_[line_id_2][0], self.args)
			# cv_img_1 = occ_augImage(cv_img_1)
			# cv_img_2 = occ_augImage(cv_img_2)
			# cv_img_1 = transform(cv_img_1, self.args)
			# cv_img_2 = transform(cv_img_2, self.args)
			cv_img_1 = cv_img_1.astype(np.float32)
			cv_img_2 = cv_img_2.astype(np.float32)
			prefetch_data.append(cv_img_1)
			prefetch_data.append(cv_img_2)
			prefetch_label.append(self.lines_[line_id_1][1])
			prefetch_label.append(self.lines_[line_id_2][1])
			if len(prefetch_data)==self.batch_size*2:
				break
			
		return np.array(prefetch_data), np.array(prefetch_label)

	# def next_epoch(self):
	# 	random.shuffle(self.sec_labels_)
	# 	this_epoch_labels = []
	# 	file_labels = []

	# 	for day in self.sec_labels_:
	# 		leaf_labels = self.ShuffleLeafsWithSecLabels(day)
	# 		num_classes = len(leaf_labels)

	# 		for item_id  in range(num_classes):
  	# 			# get a batch data
	# 		  	class_label = leaf_labels[item_id]
	# 		  	if class_label not in this_epoch_labels:
	# 		  		lines = self.class_lines_map_[day][class_label]
	# 		  		idx_1 = random.randint(0, len(lines)) % len(lines)
	# 		  		idx_2 = random.randint(0, len(lines)) % len(lines)
	# 		  		line_id_1 = lines[idx_1]
	# 		  		line_id_2 = lines[idx_2]
	# 		  		this_epoch_labels.append(class_label)
					
	# 		  		file_labels.append([[self.lines_[line_id_1][0], self.lines_[line_id_2][0]], self.lines_[line_id_1][1]])

	# 	return file_labels
