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
import pdb
import math
import random
from network import resNet50
# from utils import get_train_dataflow
# from input_source import TFDatasetInput

num_to_str = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f'}

def preprocessing(filename):
    height = 384
    width = 128
    is_color = True
    mirror = True and np.random.randint(0, 2)
    mean_values_ = [104, 117, 123]

    cv_img = None
    # cv_img_origin = tf.io.decode_image(tf.io.read_file(filename), channels=3) # get rgb data
    # cv_img_origin = cv_img_origin.numpy()
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
        
        # for i in range(100):
        area = height * width
        scale = np.random.uniform(sl, sh)
        target_area = scale * area
        aspect_ratio = np.random.uniform(as_r, 1/as_r)
        h = int(math.sqrt(target_area * aspect_ratio))
        w = int(math.sqrt(target_area / aspect_ratio))
        x1 = 0
        x2 = 0
        if w < width and h < height:
            x1 = np.random.randint(0, height - h)
            x2 = np.random.randint(0, width - w)
            cv_img[x1: x1 + h, x2: x2 + w] = np.random.randint(0, 255, (h, w, 3))
            #break

    if len(mean_values_)>0:
        cv_img = cv_img - np.array(mean_values_).reshape([1,1,3])

    cv_img = cv_img.astype(np.float32)

    return cv_img




def main(logger, args):
    NpairData = npair_data.NpairDataClass(args)
    NpairData.setUp()
    model = resNet50([3, 4, 6, 3], [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]])
    model.build(input_shape=(None, 384, 128, 3))

    optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)

    # load pre-trained model by caffe
    restore_from_caffe.load(model)
    
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, args.model_path, max_to_keep=1)
    # ckpt.restore(manager.latest_checkpoint)
    # if manager.latest_checkpoint:
    #     print("Restored from {}".format(manager.latest_checkpoint))
    # else:
    #     print("Initializing from scratch.")

    # tf.saved_model.save(
    #   model, './saved_model/', signatures=model.call.get_concrete_function(
    #       tf.TensorSpec(shape=[None, 384, 128, 3], dtype=tf.float32, name="inp")))

    iter_count = optimizer.iterations.numpy()
    tvars = model.trainable_variables

    def map_data(filename, label):
        image = tf.py_function(preprocessing, inp=[filename], Tout=tf.float32)
        image.set_shape([args.height, args.width, args.channels])

        return image, label

    while iter_count<args.max_iters:
        files, labels = NpairData.next_iteration()
        files = tf.convert_to_tensor(files)
        labels = tf.convert_to_tensor(labels)
        dataset = tf.data.Dataset.from_tensor_slices((files, labels)).map(map_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(args.batch*2, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        # for images, labels in dataset:
        # images, labels = NpairData.prefetchData()
        # file_labels = NpairData.next_iteration()
        # dataset = get_train_input(file_labels, args)
        for images, labels in dataset:
            with tf.GradientTape() as tape:
                iter_count += 1
                images = tf.reshape(images, [-1, args.height, args.width, 3])
                labels = tf.reshape(labels, [-1])

                logits = model(images, training=False)

                loss = npair_loss(labels, logits, topK=args.topk)
                # regularization_loss = tf.math.add_n(model.losses)
                total_loss = loss

            grads = tape.gradient(total_loss, tvars)
            # grads[-1] = grads[-1] * 2
            optimizer.apply_gradients(zip(grads, tvars))

            if iter_count % 100 == 0:
                logger.info("iters: " + str(iter_count) + 
                            " npair_loss: " + str(loss.numpy()) + 
                            " batch: " + str(args.batch*2) + 
                            " lr: " + str(optimizer.learning_rate.numpy()))
                save_path = manager.save()
                if iter_count % 10000 == 0:
                    ckpt.save('./model/store/iter-'+str(iter_count))
            if iter_count==240000:
                optimizer.learning_rate = args.lr/10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--logfile", default='./trian.log', type=str)

    parser.add_argument("--height", default=384)

    parser.add_argument("--width", default=128)

    parser.add_argument("--channels", default=3)

    parser.add_argument("--is_color", default=True)

    parser.add_argument("--crop_size", default=0)

    parser.add_argument("--batch", default=12)

    parser.add_argument("--iterations_per_epoch", default=10)

    parser.add_argument("--mean_values_", default=[104, 117, 123])

    parser.add_argument("-root_folder", default="/ssd/data/")

    parser.add_argument("--source", default="/ssd/train.txt") # codes/bnneck_npair_256_mega_80_lr_dropout_nor

    parser.add_argument("--model_path", default='./model/')

    parser.add_argument("--pre_trained_model", default=None)

    parser.add_argument("--topk", default=10)

    parser.add_argument("--phase", default="Train")

    parser.add_argument("--occ_aug", default=True)

    parser.add_argument("--shuffle", default=True)

    parser.add_argument("--max_iters", default=320000)

    parser.add_argument("--lr", default=0.0001)

    parser.add_argument("--process_nums", default=40)

    parser.add_argument("--mirror", default=True)

    parser.add_argument("--scale", default=1.0)

    parser.add_argument("--weight_decay", default=0.00005)


    args = parser.parse_args()


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(args.logfile, mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(physical_devices[7], 'GPU')

    if tf.__version__.startswith('2.'):
        tf.config.set_soft_device_placement(True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    # logical_devices = tf.config.experimental.list_logical_devices('GPU')
    # assert len(logical_devices) == len(physical_devices) - 6
    main(logger, args)
    # im = preprocessing("C:\\Users\\xukaiping\\Desktop\\ch02002_20181031103725_pano_00000621_00001800.jpg")
    # print(im)
