import tensorflow as tf
from tensorflow import keras
from npair_loss import npair_loss
import npair_data
import logging
import cv2
import numpy as np
import time
import argparse
import restore_from_caffe
import math
import random

num_to_str = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f'}

def preprocessing(filename):
    height = 384
    width = 128
    mirror = np.random.randint(0, 2)
    mean_values_ = [104, 117, 123]

    cv_img = None
    cv_img_origin = cv2.imread(filename.numpy().decode(), cv2.IMREAD_COLOR)

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

    if mirror:
        cv_img = cv2.flip(cv_img, 1)

    prob = 0.5
    this_prob = np.random.uniform(0, 1)
    if this_prob > prob:
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
                cv_img[x1: x1 + h, x2: x2 + w] = np.random.randint(0, 255, (h, w, 3))
                break

    if len(mean_values_)>0:
        cv_img = cv_img - np.array(mean_values_).reshape([1,1,3])

    cv_img = cv_img.astype(np.float32)

    return cv_img

class resnetBlock(keras.Model):
    def __init__(self, out_channels, block_id, layer_id, residual_path=False, down_sampling=False):
        super(resnetBlock, self).__init__()
        self.out_channels = out_channels
        self.residual_path = residual_path

        conv1_name = "res" + str(block_id+2) + num_to_str[layer_id] + '_branch2a'
        if down_sampling:
            self.conv1 = keras.layers.Conv2D(self.out_channels[0], (1, 1), 2, padding='same', use_bias=False, 
                                        kernel_regularizer=tf.keras.regularizers.l2(0.00005), name=conv1_name)
        else:
            self.conv1 = keras.layers.Conv2D(self.out_channels[0], (1, 1), 1, padding='same', use_bias=False, 
                                        kernel_regularizer=tf.keras.regularizers.l2(0.00005), name=conv1_name)

        self.bn1 = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, beta_regularizer=tf.keras.regularizers.l2(0.00005), gamma_regularizer=tf.keras.regularizers.l2(0.00005), trainable=True, name="bn"+conv1_name[3:])
        self.relu1 = keras.layers.ReLU(name=conv1_name+"_relu")

        conv2_name = "res" + str(block_id+2) + num_to_str[layer_id] + '_branch2b'
        self.conv2 = keras.layers.Conv2D(self.out_channels[1], (3, 3), 1, padding='same', use_bias=False, 
                                        kernel_regularizer=tf.keras.regularizers.l2(0.00005), name=conv2_name)
        self.bn2 = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, beta_regularizer=tf.keras.regularizers.l2(0.00005), gamma_regularizer=tf.keras.regularizers.l2(0.00005), trainable=True, name="bn"+conv2_name[3:])
        self.relu2 = keras.layers.ReLU(name=conv2_name+"_relu")

        conv3_name = "res" + str(block_id+2) + num_to_str[layer_id] + '_branch2c'
        self.conv3 = keras.layers.Conv2D(self.out_channels[2], (1, 1), 1, padding='same', use_bias=False, 
                                        kernel_regularizer=tf.keras.regularizers.l2(0.00005), name=conv3_name)
        self.bn3 = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, beta_regularizer=tf.keras.regularizers.l2(0.00005), gamma_regularizer=tf.keras.regularizers.l2(0.00005), trainable=True, name="bn"+conv3_name[3:])

        residual_name = "res" + str(block_id+2) + num_to_str[layer_id] + '_branch1'
        if residual_path and down_sampling:
            self.down_conv = keras.layers.Conv2D(self.out_channels[2], (1, 1), 2, padding='same', use_bias=False, 
                                                kernel_regularizer=tf.keras.regularizers.l2(0.00005), name=residual_name)
        elif residual_path:
            self.down_conv = keras.layers.Conv2D(self.out_channels[2], (1, 1), 1, padding='same', use_bias=False, 
                                                kernel_regularizer=tf.keras.regularizers.l2(0.00005), name=residual_name)

        self.down_bn = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, beta_regularizer=tf.keras.regularizers.l2(0.00005), gamma_regularizer=tf.keras.regularizers.l2(0.00005), trainable=True, name="bn"+residual_name[3:])

        self.relu3 = keras.layers.ReLU(name="res" + str(block_id+2) + num_to_str[layer_id]+"_relu")

    def call(self, inputs, training = True):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        if self.residual_path:
            residual = self.down_conv(residual)
            residual = self.down_bn(residual, training=training)
        x = x + residual
        x = self.relu3(x)
        return x

class resNet50(keras.Model):
    def __init__(self, block_list, out_channels, **kwargs):
        super(resNet50, self).__init__(**kwargs)
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.out_channels = out_channels

        self.conv1 = keras.layers.Conv2D(64, (7, 7), 2, padding=[[0,0], [3,2], [3,2], [0,0]], use_bias=True, 
                                        kernel_regularizer=tf.keras.regularizers.l2(0.00005), 
                                        bias_regularizer=tf.keras.regularizers.l2(0.00005), name='conv1')
        self.bn_conv1 = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, beta_regularizer=tf.keras.regularizers.l2(0.00005), gamma_regularizer=tf.keras.regularizers.l2(0.00005), trainable=True, name='bn_conv1')
        self.conv1_relu = keras.layers.ReLU(name="con1_relu")
        self.pool1 = keras.layers.MaxPool2D((3, 3), 2, padding='same', name='bn_conv1')

        self.blocks = keras.models.Sequential()
        # build all the blocks
        for block_id in range(self.num_blocks):
            for layer_id in range(block_list[block_id]):
                if (block_id==0 or block_id==3) and layer_id==0:
                    block = resnetBlock(self.out_channels[block_id], block_id, layer_id, residual_path=True, down_sampling=False)
                elif block_id!=0 and layer_id == 0:
                    block = resnetBlock(self.out_channels[block_id], block_id, layer_id, residual_path=True, down_sampling=True)
                else:
                    block = resnetBlock(self.out_channels[block_id], block_id, layer_id, residual_path=False, down_sampling=False)
                self.blocks.add(block)

        self.pool5 = keras.layers.MaxPool2D((24, 8), 1, name="pool5")
        self.drop_fc = keras.layers.Dropout(0.5)
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.00005), name='fc256')

    def call(self, inputs, training = False):
        out = self.conv1(inputs)
        out = self.bn_conv1(out, training=training)
        out = self.conv1_relu(out)
        out = self.pool1(out)

        out = self.blocks(out, training=training)

        out = self.pool5(out)
        out = self.drop_fc(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = 200 * tf.math.l2_normalize(out, axis=1, epsilon=1e-12)

        return out


def custom_loop(strategy, args):
    assert strategy is not None
    assert tf.executing_eagerly()
    NpairData = npair_data.NpairDataClass(args)
    NpairData.setUp()

    with strategy.scope():
        assert tf.executing_eagerly()
        
        optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
        model = resNet50([3, 4, 6, 3], [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]])
        model.build(input_shape=(None, 384, 128, 3))
        
        # load pre-trained model by caffe
        restore_from_caffe.load(model)

        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, args.model_dir, max_to_keep=1)
        # ckpt.restore(manager.latest_checkpoint)
        # if manager.latest_checkpoint:
        #     print("Restored from {}".format(manager.latest_checkpoint))
        # else:
        #     print("Initializing from scratch.")

        tvars = model.trainable_variables
        def train_step(iterator):
            """Performs a distributed training step."""
            def _distributed_train_step(inputs):
                """Replicated training step."""
                with tf.GradientTape() as tape:
                    images, labels = inputs
                    images = tf.reshape(images, [-1, args.height, args.width, args.channels])
                    labels = tf.reshape(labels, [-1])
                    logits = model(images)
                    
                    loss = npair_loss(labels, logits, topK=args.topk)
                    loss = loss / len(args.gpus)
                    # regularization_loss = tf.math.add_n(model.losses) / len(args.gpus)
                    total_loss = loss # + regularization_loss
                   
                grads = tape.gradient(total_loss, tvars)
            
                optimizer.apply_gradients(zip(grads, tvars))

                return loss

            per_replica_losses  = strategy.experimental_run_v2(
                _distributed_train_step, args=(iterator,))

            # for report
            loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            return loss

        def map_data(filename, label):
            image = tf.py_function(preprocessing, inp=[filename], Tout=tf.float32)
            image.set_shape([args.height, args.width, args.channels])

            return image, label

        iter_count = optimizer.iterations.numpy()
        while iter_count<args.max_iters:
            files, labels = NpairData.next_iteration()

            files = tf.convert_to_tensor(files)
            labels = tf.convert_to_tensor(labels)

            dataset = tf.data.Dataset.from_tensor_slices((files, labels))
            dataset = dataset.map(map_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(args.batch*2, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
            train_iterator = strategy.experimental_distribute_dataset(dataset)

            for train_data in train_iterator:
                iter_count += 1
                loss = train_step(train_data)
                if iter_count % 100 == 0:
                    print("iters: " + str(iter_count) + 
                                " npair_loss: " + str(loss.numpy()) + 
                                " batch: " + str(args.batch*2) + 
                                " lr: " + str(optimizer.learning_rate.numpy()) +
                                " time: " + time.asctime(time.localtime(time.time())))
                    save_path = manager.save()
                    if iter_count % 10000 == 0:
                        ckpt.save('./model/store/iter-'+str(iter_count))
                if iter_count==60000:
                    optimizer.learning_rate = args.lr/10

def main(args):
    device = ["device:GPU:%d" % i for i in args.gpus]
    strategy = tf.distribute.MirroredStrategy(
            devices=device,
            cross_device_ops=tf.distribute.NcclAllReduce(num_packs=1))

    print(device)
    if strategy:
        # flags_obj.enable_get_next_as_optional controls whether enabling
        # get_next_as_optional behavior in DistributedIterator. If true, last
        # partial batch can be supported.
        strategy.extended.experimental_enable_get_next_as_optional = (False)
    custom_loop(strategy, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", default='./trian.log', type=str)

    parser.add_argument("--height", default=384)

    parser.add_argument("--width", default=128)

    parser.add_argument("--channels", default=3)

    parser.add_argument("--is_color", default=True)

    parser.add_argument("--crop_size", default=0)

    parser.add_argument("--batch", default=12*8)

    parser.add_argument("--gpus", default=[0, 1, 2, 3, 4, 5, 6, 7])

    parser.add_argument("--iterations_per_epoch", default=1000)

    parser.add_argument("--mean_values_", default=[104, 117, 123])

    parser.add_argument("--root_folder", default="/ssd/data/")

    parser.add_argument("--source", default="/ssd/train.txt")

    parser.add_argument("--model_dir", default='./model/')

    parser.add_argument("--topk", default=10)

    parser.add_argument("--phase", default="Train")

    parser.add_argument("--occ_aug", default=True)

    parser.add_argument("--shuffle", default=True)

    parser.add_argument("--max_iters", default=80000)

    parser.add_argument("--lr", default=0.0001)

    parser.add_argument("--mirror", default=True)

    parser.add_argument("--scale", default=1.0)


    args = parser.parse_args()
    
    # train environment config
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    tf.config.experimental.set_visible_devices(physical_devices[0:], 'GPU')
    # logical_devices = tf.config.experimental.list_logical_devices('GPU')
    # print(logical_devices)
    # assert len(logical_devices) == len(physical_devices) - 4
    print("visiable device: ", tf.config.experimental.get_visible_devices())

    main(args)