import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, BatchNormalization,Concatenate, AveragePooling3D, Activation
from tensorflow.keras.optimizers import Adam
from config import *

def conv_bn_relu(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def inception_base(x):
    x = conv_bn_relu(x, filters=32)
    x = conv_bn_relu(x, filters=32)
    x = conv_bn_