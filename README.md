
# 3DCNN-Models-Tensorflow2

This repository is a comprehensive collection of 3D implementations of commonly used 2D CNNs. It fills a significant gap as the majority of deep learning libraries offer ready-to-use 2D models but 3D models are generally less accessible.

## Required Libraries:
```ruby
* Python 3.x 
* Tensorflow 2.X
* Numpy
* random
```

## Available 3D Models:

### Classification:
```ruby
* Resnet_3D.py 
* DenseNet_3D.py
* VGG_3D.py
* Inception_3D.py
```
### Segmentation:
```ruby
* DenseVnet3D.py 
* Unet3D.py
```
### Execution Steps
Customize the models based on your requirements and the capabilities of your GPUs using *config.py*:
```ruby

##-----Network Configuration----#####
NUMBER_OF_CLASSES=5
INPUT_PATCH_SIZE=(96,128,128, 1)
##------Resnet3D----####
TRAIN_NUM_RES_UNIT=3
TRAIN_NUM_FILTERS=(16, 32, 64, 128)
TRAIN_STRIDES=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2))
TRAIN_CLASSIFY_ACTICATION=tf.nn.relu6
TRAIN_KERNAL_INITIALIZER=tf.keras.initializers.VarianceScaling(distribution='uniform')
#-------DenseNet13D----#####
# DenseNet
DENSE_NET_BLOCKS = 3
DENSE_NET_BLOCK_LAYERS = 5
DENSE_NET_INITIAL_CONV_DIM = 16
DENSE_NET_GROWTH_RATE = DENSE_NET_INITIAL_CONV_DIM // 2
DENSE_NET_ENABLE_BOTTLENETCK = False # called DenseNet-BC if ENABLE_BOTTLENETCK and COMPRESSION < 1 in paper
DENSE_NET_TRANSITION_COMPRESSION = 1.0
DENSE_NET_ENABLE_DROPOUT = True
DENSE_NET_DROPOUT = 0.5
#-------Inception3D----#####
INCEPTION_BLOCKS = 6
INCEPTION_REDUCTION_STEPS = 2
INCEPTION_KEEP_FILTERS = 128
INCEPTION_ENABLE_DEPTHWISE_SEPARABLE_CONV_SHRINKAGE = 0.333
INCEPTION_ENABLE_SPATIAL_SEPARABLE_CONV = True
INCEPTION_DROPOUT = 0.5
#---------VGG3D----####
TRAIN_CLASSIFY_USE_BN = False

```