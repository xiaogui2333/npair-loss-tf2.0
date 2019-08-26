import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from input_source import TFDatasetInput
from utils import get_test_dataflow
import time
import argparse
import restore_from_caffe
import os
import pr
import metric_compute
from metric_compute import Same_Pair_Requirements

num_to_str = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f'}

def get_test_input(data_dir, args):
    ds = get_test_dataflow(data_dir, args)
    ds = TFDatasetInput.dataflow_to_dataset(ds, (tf.float32, tf.int32),
                                                ([None, None, None],
                                                 []))
    test_input_dataset = ds.batch(args.batch, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
    return test_input_dataset

class resnetBlock(keras.Model):
    def __init__(self, out_channels, block_id, layer_id, residual_path=False, down_sampling=False):
        super(resnetBlock, self).__init__()
        self.out_channels = out_channels
        self.residual_path = residual_path

        conv1_name = "res" + str(block_id+2) + num_to_str[layer_id] + '_branch2a'
        if down_sampling:
            self.conv1 = keras.layers.Conv2D(self.out_channels[0], (1, 1), 2, padding='same', use_bias=False, name=conv1_name)
        else:
            self.conv1 = keras.layers.Conv2D(self.out_channels[0], (1, 1), 1, padding='same', use_bias=False, name=conv1_name)
        self.bn1 = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, name="bn"+conv1_name[3:])
        self.relu1 = keras.layers.ReLU(name=conv1_name+"_relu")

        conv2_name = "res" + str(block_id+2) + num_to_str[layer_id] + '_branch2b'
        self.conv2 = keras.layers.Conv2D(self.out_channels[1], (3, 3), 1, padding='same', use_bias=False, name=conv2_name)
        self.bn2 = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, name="bn"+conv2_name[3:])
        self.relu2 = keras.layers.ReLU(name=conv2_name+"_relu")

        conv3_name = "res" + str(block_id+2) + num_to_str[layer_id] + '_branch2c'
        self.conv3 = keras.layers.Conv2D(self.out_channels[2], (1, 1), 1, padding='same', use_bias=False, name=conv3_name)
        self.bn3 = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, name="bn"+conv3_name[3:])

        residual_name = "res" + str(block_id+2) + num_to_str[layer_id] + '_branch1'
        if residual_path and down_sampling:
            self.down_conv = keras.layers.Conv2D(self.out_channels[2], (1, 1), 2, padding='same', use_bias=False, name=residual_name)
        elif residual_path:
            self.down_conv = keras.layers.Conv2D(self.out_channels[2], (1, 1), 1, padding='same', use_bias=False, name=residual_name)
        self.down_bn = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, name="bn"+residual_name[3:])

        self.relu3 = keras.layers.ReLU(name="res" + str(block_id+2) + num_to_str[layer_id]+"_relu")

    def call(self, inputs, training = False):
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
    def __init__(self, block_list, out_channels, training, **kwargs):
        super(resNet50, self).__init__(**kwargs)
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.out_channels = out_channels

        self.conv1 = keras.layers.Conv2D(64, (7, 7), 2, padding=[[0,0], [3,2], [3,2], [0,0]], use_bias=True, name='conv1')
        self.bn_conv1 = keras.layers.BatchNormalization(epsilon=1e-5, center=True, scale=True, name='bn_conv1')
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

        out = self.flatten(out)
        out = self.fc(out)
        out = 200 * tf.math.l2_normalize(out, axis=1, epsilon=1e-12)
        return out

def get_batch(lines, iters, batch):
    batch_files_labels = [[lines[i].split()[0], os.path.basename(lines[i].split()[0])] for i in range(batch*iters, batch*(iters+1)) if i<len(lines)]
    return batch_files_labels

def extract_feature(strategy, args):
    assert strategy is not None
    assert tf.executing_eagerly()

    with open(args.evaluate_data, 'r') as fio:
        lines = fio.read().splitlines()
    print("total files: ", len(lines))

    start_index = 0
    file_paths_labels = [[line.split()[0], idx] for idx, line in enumerate(lines)]
    datasets = get_test_input(file_paths_labels[start_index:], args)

    test_data = strategy.experimental_distribute_dataset(datasets)

    with strategy.scope():
        assert tf.executing_eagerly()
        model = resNet50([3, 4, 6, 3], [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]], False)
        optimizer = keras.optimizers.SGD()
        model.build(input_shape=(None, args.height, args.width, args.channels))

        # load well-trained model
        if args.load_caffemodel:
            restore_from_caffe.load(model)
        else:
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
            manager = tf.train.CheckpointManager(ckpt, args.model_path, max_to_keep=1)
            ckpt.restore(manager.latest_checkpoint).expect_partial()
            if manager.latest_checkpoint:
                print("restored from {}".format(manager.latest_checkpoint))
            else:
                raise Exception("there is no model file in folder '" + args.model_path + "' or missing file checkpoint")

        # @tf.function
        def test_step(iterator):
            def _test_step_fn(inputs):
                images, line_id = inputs
                if images==None or images.shape[0]==0:
                    print("current gpu has 0 images")
                    return
                logits = model(images, training=False)
                for i in range(logits.shape[0]):
                    feat_path = os.path.join(args.feat_fold, os.path.basename(file_paths_labels[line_id[i]][0])+ '.npy')
                    np.save(feat_path, logits[i].numpy())
                return

            strategy.experimental_run_v2(_test_step_fn, args=(iterator,))
            
        count = start_index
        start_time = time.time()
        for data in test_data:
            test_step(data)
            count += args.batch
            if count%100==0:
                print("get features: ", count)
        print("total features: ", count, "times: ", time.time()-start_time, ' secs')

    del strategy

def evaluation(agrs):
    pr.init(3)

    same_pair_requirements = Same_Pair_Requirements(min_frame_interval=args.min_frame_interval, max_frame_interval=args.max_frame_interval, must_same_day=args.must_same_day, must_different_days=args.must_different_days, must_same_camera=args.must_same_camera,
                                                    must_diff_camera=args.must_diff_camera, must_same_video=args.must_same_video, must_diff_video=args.must_diff_video, same_sample_size=args.same_sample_size)
    val_list_path = args.test_file_path
    feat_fold = args.feat_fold

    # load dataset
    with open(val_list_path, 'r') as fio:
        lines = fio.read().splitlines()
    num_line = len(lines)

    pr.in_loop('extract feature', num_line, .1)

    gt_ids = []
    gt_counts = {}
    feats = []
    features_per_person_dict = {}
    crops_file_list_dict = {}

    for line_id in range(num_line):
        pr.loop(line_id)
        # image path
        parts = lines[line_id].split()
        feat_path = os.path.join(feat_fold, parts[0].split('/')[-1] + '.npy')
        gt_id = int(parts[1])

        # load feature
        feat = np.load(feat_path)

        try:
            features_per_person_dict[gt_id].append(feat)
            crops_file_list_dict[gt_id].append(parts[0])
        except:
            features_per_person_dict[gt_id] = [feat]
            crops_file_list_dict[gt_id] = [parts[0]]

    pr.out_loop(num_line)

    features_per_person = []
    crops_file_list = []

    # pid order, a little bit affects the test results. This order may not be the original order
    pid = [30730, 30776, 6225, 30823, 30825, 30833, 149668, 258222, 30925, 30926, 30927, 350438, 31005, 30428, 379, 233927, 350784, 43585, 8788, 10842, 262877, 230110, 140014, 347621, 30190, 355834, 37858, 283621, 355332, 30556, 30567, 351410, 140479, 356214, 347681, 355888, 355642, 167251, 5487, 355752, 355753, 355754, 355755, 355756, 355757, 355758, 355759, 355760, 355761, 355762, 355763, 355764, 355765, 355766, 355767, 355768, 355770, 355771, 355772, 355773, 355774, 355775, 355776, 355777, 355778, 355780, 355781, 355782, 355783, 355784, 355785, 355787, 355788, 355789, 355790, 355792, 355793, 355794, 355795, 355796, 355797, 355798, 355799, 355800, 355802, 355803, 355804, 355805, 355806, 355807, 355808, 355809, 355810, 355811, 355812, 355813, 355814, 355815, 355817, 355818, 355819, 355820, 347629, 355822, 355823, 355825, 355826, 355827, 355828, 355830, 355831, 355832, 355833, 257530, 355835, 257532, 355837, 355838, 355839, 355840, 355841, 355842, 355843, 355844, 355845, 355846, 355847, 355849, 355850, 355851, 355852, 355853, 355854, 355855, 355856, 355857, 355858, 355859, 355860, 355861, 355862, 355863, 355864, 355865, 355867, 355868, 355869, 355870, 355871, 355872, 355873, 355874, 355875, 355876, 355877, 355878, 355879, 355880, 355881, 355882, 355883, 355884, 355885, 355886, 355887, 17968, 355889, 355891, 355893, 355895, 355896, 355897, 355899, 355902, 355903, 355904, 355906, 355907, 355908, 355909, 355910, 355911, 355912, 355914, 355915, 355917, 355918, 355919, 355920, 355921, 355922, 355923, 355924, 355925, 355926, 355927, 355928, 355929, 355930, 355931, 355932, 355933, 355934, 355935, 355936, 355937, 355940, 355941, 355942, 355943, 355944, 355945, 355946, 355948, 355951, 355952, 355953, 355954, 355955, 355957, 355958, 355959, 355960, 355961, 355962, 355963, 355964, 355965, 355966, 355967, 355968, 355969, 355970, 355971, 355972, 355973, 355974, 355975, 355976, 355977, 355978, 355979, 355980, 355981, 355982, 355983, 355984, 355985, 355986, 355988, 355989, 355990, 355991, 355992, 355993, 355994, 355995, 355996, 355998, 355999, 356000, 356001, 356002, 356004, 356005, 356006, 356007, 356008, 356009, 356010, 356011, 356012, 356013, 356014, 356015, 356017, 356018, 356019, 356020, 356021, 356022, 356024, 356025, 356026, 356027, 356028, 7869, 356030, 356031, 356033, 356034, 356035, 356036, 356037, 356038, 356039, 356040, 356043, 356044, 356045, 356046, 356047, 356048, 356049, 356050, 356051, 356052, 356053, 356054, 356056, 356057, 356058, 356059, 356060, 356062, 356063, 356065, 356066, 356067, 356070, 356071, 356073, 356074, 356075, 356076, 356078, 356079, 356080, 356081, 356084, 356085, 356086, 356087, 356088, 356089, 356090, 356091, 356093, 356094, 356095, 356096, 356097, 356098, 356099, 356100, 356101, 356103, 356104, 356105, 356106, 356107, 356108, 356109, 356110, 356111, 356112, 356114, 356115, 356116, 356117, 356118, 356119, 356120, 356121, 356122, 356123, 356125, 356126, 356127, 356128, 356129, 356130, 356131, 356132, 356133, 356134, 356135, 356136, 356137, 356138, 356139, 356141, 356142, 356143, 356144, 356145, 356146, 356147, 356148, 356149, 356150, 356151, 356152, 356153, 356154, 356155, 356156, 356157, 356158, 356159, 356161, 356162, 356163, 356164, 356165, 356166, 356167, 356169, 356170, 356172, 356173, 356174, 356175, 356176, 356177, 356179, 356180, 356181, 356182, 356184, 356185, 356186, 356187, 356188, 356189, 356191, 356192, 356193, 356194, 356195, 356196, 356197, 356198, 356199, 356201, 356202, 356203, 356204, 356205, 356206, 356207, 356208, 356209, 356210, 356211, 356212, 356213, 30582, 356215, 356216, 356217, 356218, 356219, 356220, 356221, 356223, 356224, 356225, 356226, 356227, 356228, 356229, 356230, 356231, 356232, 356233, 30606, 24476]
    for gt_id in pid:
        features_per_person.append(features_per_person_dict[gt_id])
        crops_file_list.append(crops_file_list_dict[gt_id])

    mean_len = int(sum([len(crop_files) for crop_files in crops_file_list]) / max(1,len(crops_file_list)))
    len_limit = int(mean_len*1.5)

    for i, crop_files in enumerate(crops_file_list):
        if len(crop_files) > len_limit:
            sample_ids = np.round(np.linspace(0, len(crop_files)-1, len_limit)).astype(int)
            crops_file_list[i] = np.array(crop_files)[sample_ids]
            features_per_person[i] = np.array(features_per_person[i])[sample_ids, :]
        else:
            crops_file_list[i] = np.array(crop_files)
            features_per_person[i] = np.array(features_per_person[i])

    print("finish feature computing")
    same_pair_dist, same_pair_files = metric_compute.compute_same_pair_dist(features_per_person, crops_file_list, same_pair_requirements)
    print("same pair dist are done") 
    diff_pair_dist, diff_pair_files = metric_compute.compute_diff_pair_dist(features_per_person, crops_file_list, folder_sample_interval=args.neg_folder_interval)
    print("diff pair dist are done")

    same_pair_dist = np.array(same_pair_dist)
    diff_pair_dist = np.array(diff_pair_dist)
    tpr2, fpr2, th2 = metric_compute.report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.01)
    tpr3, fpr3, th3 = metric_compute.report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.001)
    tpr4, fpr4, th4 = metric_compute.report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.0001)
    tpr5, fpr5, th5 = metric_compute.report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.00001)

    print('same_pairs are {0}, diff_pairs are {1}'.format(str(same_pair_dist.size), str(diff_pair_dist.size)))
    print('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'.format('%.3f'%tpr2, '%.6f'%th2, '%.5f'%fpr2, args.job_name, 'dsc'))
    print('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'.format('%.3f'%tpr3, '%.6f'%th3, '%.5f'%fpr3, args.job_name, 'dsc'))
    print('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'.format('%.3f' % tpr4, '%.6f'%th4, '%.5f' % fpr4, args.job_name, 'dsc'))
    print('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'.format('%.3f' % tpr5, '%.6f' % th5, '%.5f' % fpr5, args.job_name, 'dsc'))

def main(args):
    gpu_list = args.gpus
    device = ["device:GPU:%d" % i for i in gpu_list]
    strategy = tf.distribute.MirroredStrategy(
            devices=device,
            cross_device_ops=None)

    if strategy:
        strategy.extended.experimental_enable_get_next_as_optional = (True)

    extract_feature(strategy, args)
    
    evaluation(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", default=384)

    parser.add_argument("--width", default=128)

    parser.add_argument("--channels", default=3)

    parser.add_argument("--is_color", default=True)

    parser.add_argument("--crop_size", default=0)

    parser.add_argument("--mean_values_", default=[104, 117, 123])

    parser.add_argument("--batch", default=32*2)

    parser.add_argument("--load_caffemodel", default=False)

    parser.add_argument("--model_path", default='./model/')
    
    parser.add_argument("--gpus", default=[5, 6])

    parser.add_argument("--phase", default="Test")

    parser.add_argument("--occ_aug", default=False)

    parser.add_argument("--shuffle", default=False)

    parser.add_argument("--process_nums", default=40)

    parser.add_argument("--mirror", default=False)

    parser.add_argument("--scale", default=1.0)

    parser.add_argument("--feat_fold", default='./feature')

    parser.add_argument("--evaluate_data", default='./eval.txt')

    parser.add_argument('--job_name', type=str, default='',
                        help='job_name')

    parser.add_argument('--test_file_path', type=str, default='./eval.txt',
                        help='test_file_name')
    parser.add_argument('--min_frame_interval', type=int, default=-1,
                        help='the min frame intervals between a same pair')

    parser.add_argument('--max_frame_interval', type=int, default=-1,
                        help='the max frame intervals between a same pair')

    parser.add_argument('--must_same_day', action='store_true', default=True,
                        help='same pair must be at same day')

    parser.add_argument('--must_different_days', action='store_true', default=False,
                        help='same pair must be at different days')

    parser.add_argument('--must_same_camera', action='store_true', default=False,
                        help='same pair must be at same cameras')

    parser.add_argument('--must_diff_camera', action='store_true', default=True,
                        help='same pair must be at different cameras')

    parser.add_argument('--must_same_video', action='store_true', default=False,
                        help='same pair must be at same video')

    parser.add_argument('--must_diff_video', action='store_true', default=False,
                        help='same pair must be at different videos')

    parser.add_argument('--same_sample_size', type=int, default=128,
                        help='sample size of in each class of same')

    parser.add_argument('--neg_folder_interval', type=int, default=5,
                    help='interval of neg sample folder')

    args = parser.parse_args()

    # config gpus environment
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    gpus = [physical_devices[i] for i in range(8)]
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    # logical_devices = tf.config.experimental.list_logical_devices('GPU')
    # assert len(logical_devices) == len(physical_devices) - 6
    main(args)
