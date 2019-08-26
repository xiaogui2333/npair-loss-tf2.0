import tensorflow as tf
import math
from tensorflow import keras
import numpy as np
import time
import pdb

# def calc_loss(batch_size, labels, qry_pos_dist, DistMatrix, mask):
# 	loss = tf.constant(0, dtype=tf.float32)
# 	for qry_idx in range(batch_size):
# 		qry_label = labels[qry_idx]
# 		ref_idx = reference_labels_idx[int(qry_label)]
# 		assert len(ref_idx)==2, "sample error!, must two sample per class."
# 		pos_idx = ref_idx[0] if ref_idx[0] != qry_idx else ref_idx[1]
# 		qry_loss = DistMatrix[qry_idx] - DistMatrix[qry_idx][pos_idx]
# 		qry_loss = tf.math.exp(qry_loss)
# 		qry_loss = tf.reduce_sum(qry_loss[0:min(ref_idx)]) + tf.reduce_sum(qry_loss[max(ref_idx)+1:])
# 		loss = tf.math.add(loss, tf.math.log(1+qry_loss))
# 	loss = loss / batch_size
# 	return loss

# this function is inefficient, it needs to be rewritten
def calc_precision(qry_num, ref_num, reference_labels_idx, DistMatrix, labels, topK):
	precision = tf.constant(0, dtype=tf.float32)
	for qry_i in range(qry_num):
		qry_label = labels[qry_i]
		precision_one = tf.constant(0, dtype=tf.float32)
		buf = DistMatrix[qry_i]
		buf = tf.sort(buf, direction='DESCENDING')

		assert topK<ref_num, "ERROR:topk >= ref_num"

		threshold = buf[topK - 1]
		for k in range(len(reference_labels_idx[qry_label])):
			ref_idx = reference_labels_idx[qry_label][k]
			if DistMatrix[qry_i][ref_idx] > threshold:
				precision_one += 1
		if precision_one > topK:
			precision_one = topK
		precision += precision_one / topK
	precision = precision / qry_num
	return precision

def npair_loss(labels, logits, topK=2, gamma_=0.001):
	batch_size = logits.shape[0]
	labels = labels.numpy()
	assert len(set(labels))==len(labels)/2, "error sample, must two sample per class."

	qry_pos_dist = []
	dist_matrix = gamma_ * tf.matmul(logits, tf.transpose(logits))
	mask =  1.0 - np.eye(batch_size, batch_size, dtype=np.float32)
	for i in range(batch_size):
		if i%2==0:
			qry_pos_dist.append(dist_matrix[i][i+1])
			mask[i][i+1] = 0
		else:
			qry_pos_dist.append(dist_matrix[i][i-1])
			mask[i][i-1] = 0

	qry_pos_dist  = tf.reshape(qry_pos_dist, [batch_size, 1])

    # calc precision
	# precision = calc_precision(batch_size, batch_size, reference_labels_idx, DistMatrix, labels, topK)

    # calc loss
	# loss = calc_loss(batch_size, labels, qry_pos_dist, DistMatrix, mask)
	loss = tf.subtract(dist_matrix, qry_pos_dist)
	loss = tf.math.exp(loss)
	loss = tf.multiply(loss, mask)
	loss = tf.reduce_sum(loss, 1)
	loss = tf.math.log(1.0 + loss)
	loss = tf.reduce_mean(loss)

	return loss

