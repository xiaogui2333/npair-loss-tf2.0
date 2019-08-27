import numpy as np
import tensorflow as tf
from tensorflow import keras

map_dict = {'kernel':'weights', 'bias':'biases', 'gamma':'scale', 'beta':'offset', 'moving_mean':'mean', 'moving_variance':'variance'}

def load(model):
    caffemodel = 'CTF_xhm_res50_did_fid_fe_asoftmax_iter_120000.npy'
    weights = np.load(caffemodel, encoding="latin1").item()
    variables = model.variables
    for var in variables:
        name = var.name.split('/')
        name = [name[-2], name[-1][:-2]]
        assert name[0] in weights.keys()
        tf.compat.v1.assign(var, weights[name[0]][map_dict[name[1]]])
    print("load pre-trained model by caffe model: ", caffemodel)