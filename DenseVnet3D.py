
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf



##########---tf bilinear UpSampling3D
def up_sampling(input_tensor, scale):
    net = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(scale, scale), interpolation='bilinear'))(input_tensor)
    net = tf.keras.layers.Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
    net = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(scale, 1), interpolation='bilinear'))(net)
    net = tf.keras.layers.Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
    return net

#######-----Bottleneck
def Bottleneck(x, nb_filter, increase_factor=4., weight_decay=1e-4):
    inter_channel = int(nb_filter * increase_factor)
    x = tf.keras.layers.Conv3D(inter_channel, (1, 1, 1),
                               kernel_initializer='he_normal',
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(x)
    x = tf.nn.relu6(x)
    return x

#####------------>>> Convolutional Block
def conv_block(input, nb_filter, kernal_size=(3, 3, 3), dilation_rate=1,
                 bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3X3 Conv3D, optional bottleneck block and dropout
    Args:
        input: Input tensor