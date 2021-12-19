
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
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: tensor with batch_norm, relu and convolution3D added (optional bottleneck)
    '''


    x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(input)
    x = tf.nn.relu6(x)

    if bottleneck:
        inter_channel = nb_filter  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
        x = tf.keras.layers.Conv3D(inter_channel, (1, 1, 1),
                   kernel_initializer='he_normal',
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(x)
        x = tf.nn.relu6(x)

    x = tf.keras.layers.Conv3D(nb_filter, kernal_size,
               dilation_rate=dilation_rate,
               kernel_initializer='he_normal',
               padding='same',
               use_bias=False)(x)
    if dropout_rate:
        x = tf.keras.layers.SpatialDropout3D(dropout_rate)(x)
    return x

##--------------------DenseBlock-------####
def dense_block(x, nb_layers, growth_rate, kernal_size=(3, 3, 3),
                  dilation_list=None,
                  bottleneck=True, dropout_rate=None, weight_decay=1e-4,
                  return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: input tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: tensor with nb_layers of conv_block appended
    '''

    if dilation_list is None:
        dilation_list = [1] * nb_layers
    elif type(dilation_list) is int:
        dilation_list = [dilation_list] * nb_layers
    else:
        if len(dilation_list) != nb_layers:
            raise ('the length of dilation_list should be equal to nb_layers %d' % nb_layers)

    x_list = [x]

    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, kernal_size, dilation_list[i],
                          bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)
        if i == 0:
            x = cb
        else:
            x = tf.keras.layers.concatenate([x, cb], axis=-1)

    if return_concat_list:
        return x, x_list
    else:
        return x

###---------transition_block
def transition_block(input, nb_filter, compression=1.0, weight_decay=1e-4,
                       pool_kernal=(3, 3, 3), pool_strides=(2, 2, 2)):
    ''' Apply BatchNorm, Relu 1x1, Conv3D, optional compression, dropout and Maxpooling3D
    Args:
        input: input tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''


    x =tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(input)
    x = tf.nn.relu6(x)
    x = tf.keras.layers.Conv3D(int(nb_filter * compression), (1, 1, 1),
               kernel_initializer='he_normal',
               padding='same',
               use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.AveragePooling3D(pool_kernal, strides=pool_strides)(x)