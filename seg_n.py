import os
import sys

import cv2
import numpy as np

import matplotlib.pyplot as plt

import keras
import keras.backend as K
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Add
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam

def residual_block(X, filters, stage):

    '''Residual block

    Parameters:
    X: input features
    filters (int, int): number of filters in each Conv2D layer
    stage (int): residual block index

    Implementation details:
    ELU - Conv - Dropout - ELU - Conv - Scaling
    - ELU: alpha = (default) 1.0
    - Conv: 3x3
    - Dropout: rate = 0.2
    - Scaling: factor = 0.3

    Return:
    X: ouput features
    '''

    X_shortcut = X
    F1, F2 = filters
    conv_base_name = 'res_' + str(stage) + '_branch_'
    scaling_factor = 0.3

    X = ELU()(X)
    X = Conv2D(filters=F1, kernel_size=(3, 3), strides=(1, 1), padding='same', name=conv_base_name+'2_a')(X)
    X = Dropout(rate=0.2)(X)
    X = ELU()(X)
    X = Conv2D(filters=F2, kernel_size=(3, 3), strides=(1, 1), padding='same', name=conv_base_name+'2_b')(X)
    X = Lambda(lambda x: x * scaling_factor)(X)
    X = Conv2D(filters=3, kernel_size=(1, 1))(X)

    X = Add()([X, X_shortcut])

    return X

def CellDetector(input_shape=(100, 100, 3)):
    
    input_layer = Input(input_shape)


if __name__=='__main__':
    
    # ---------------------
    #  Test residual block 
    # ---------------------

    tf.reset_default_graph()
    with tf.Session() as test:
        np.random.seed(123)
        A_prev = tf.placeholder('float', [5, 100, 100, 3])
        X = np.random.randn(5, 100, 100, 3)
        A = residual_block(A_prev, filters=[3, 3], stage=1)
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
        print('\nTesting residual block:')
        print('out = ' + str(out[0][1][1][0]))
        print('Test complete: PASSED\n')

