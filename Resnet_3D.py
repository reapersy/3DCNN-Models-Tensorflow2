from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from config import*
from loss_funnction_And_matrics import*
import numpy as np

###Residual Block
def Residual_Block(inputs,
                 out_filters,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 use_bias=False,
                 activation=tf.nn.relu6,
                 kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform'),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                 bias_regularizer=None,
                 **kwargs):


    conv_params={'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    in_filters = inputs.get_shape().as_list()[-1]
    x=inputs
    orig_x=x

    ##building
    # Adjust the strided conv kernel size to prevent losing information
    k = [s * 2 if s > 1 else k for k, s in zip(kernel_size, strides)]

    if np.prod(strides) != 1:
            orig_x = tf.keras.layers.MaxPool3D(pool_size=strides,strides=strides,padding='valid')(orig_x)

    ##sub-unit-0
    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    x=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=k,strides=strides,**conv_params)(x)

    ##sub-unit-1
    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    x=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=kernel_size,strides=(1,1,1),**conv_params)(x)

        # Handle differences in input and output filter sizes
    if in_filters < out_filters:
        orig_x = tf.pad(tensor=orig_x,paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                    int(np.floor((out_filters - in_filters) / 2.)),
                    int(np.ceil((out_filters - in_filters) / 2.))]])

    elif in_filters > out_filters:
        orig_x = tf.keras.layers.Conv3D(filters=out_filters,kernel_size=kernel_size,strides=(1,1,1),**conv_param