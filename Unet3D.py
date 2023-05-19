from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, UpSampling3D
import tensorflow as tf

def Unet3D(inputs,num_classes):
    x=inputs
    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same',data_format="channels_last")(x)
    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same')(pool3