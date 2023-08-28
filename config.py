import tensorflow as tf
import math
###---Number-of-GPU

##-----Network Configuration----#####
NUMBER_OF_CLASSES=5
INPUT_PATCH_SIZE=(96,128,128, 1)
##------Resnet3D----####
TRAIN_NUM_RES_UNIT=3
TRAIN_NUM_FILTERS=(16, 32, 64, 128)
TRAIN_STRIDES=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2))
TRAIN_CLASSIFY_ACTICATION=tf.nn.relu6
TRAIN_KERNAL_INITIALIZER=tf.keras.initializers.Vari