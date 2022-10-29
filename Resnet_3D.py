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
                   'bias_r