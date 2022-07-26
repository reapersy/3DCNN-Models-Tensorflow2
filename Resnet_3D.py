from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from config import*
from loss_funnction_And_matrics import*
import numpy as np

###Residual Block
def Residual_Block(inputs,
