import tensorflow as tf
from tensorflow import keras

num_to_str = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f'}

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

    def call(self, inputs, training = True):
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