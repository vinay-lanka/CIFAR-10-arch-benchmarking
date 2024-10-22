"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    X = Img

    X = tf.layers.conv2d(inputs=X,name='conv_1',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_2',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_1',pool_size=2,strides=2,padding='same')

    X = tf.layers.conv2d(inputs=X,name='conv_3',filters=64,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_4',filters=64,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_2',pool_size=2,strides=2,padding='same')

    X = tf.layers.flatten(inputs=X)
    X = tf.layers.dense(inputs=X,name='fc_1',units=64,activation=tf.nn.relu)
    X = tf.layers.dense(inputs=X,name='fc_2',units=16,activation=tf.nn.relu)

    prLogits = tf.layers.dense(inputs=X,name='logits',units=10,activation=None)
    prSoftMax = tf.nn.softmax(prLogits,name='softmax')

    return prLogits, prSoftMax


def CIFAR10Model_optimized(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    X = Img

    X = tf.layers.conv2d(inputs=X,name='conv_1',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_2',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_1',axis=-1,center=True,scale=True)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_1',pool_size=2,strides=2,padding='same')

    X = tf.layers.conv2d(inputs=X,name='conv_3',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_4',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_2',axis=-1,center=True,scale=True)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_2',pool_size=2,strides=2,padding='same')

    X = tf.layers.conv2d(inputs=X,name='conv_5',filters=64,kernel_size=5,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_6',filters=64,kernel_size=5,padding='same',activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_3',axis=-1,center=True,scale=True)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_3',pool_size=2,strides=2,padding='same')

    X = tf.layers.conv2d(inputs=X,name='conv_7',filters=64,kernel_size=5,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_8',filters=64,kernel_size=5,padding='same',activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_4',axis=-1,center=True,scale=True)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_4',pool_size=2,strides=2,padding='same')

    X = tf.layers.flatten(inputs=X)
    X = tf.layers.dense(inputs=X,name='fc_1',units=64,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_5',axis=-1,center=True,scale=True)
    X = tf.layers.dense(inputs=X,name='fc_2',units=32,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_6',axis=-1,center=True,scale=True)
    X = tf.layers.dense(inputs=X,name='fc_3',units=16,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_7',axis=-1,center=True,scale=True)

    prLogits = tf.layers.dense(inputs=X,name='logits',units=10,activation=tf.nn.relu)
    prSoftMax = tf.nn.softmax(prLogits,name='softmax')

    return prLogits, prSoftMax

def CIFAR10Model_ResNet(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    X = Img

    X = tf.layers.conv2d(inputs=X,name='conv_1',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_2',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_1',axis=-1,center=True,scale=True)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_1',pool_size=2,strides=2,padding='same')

    X1 = tf.layers.conv2d(inputs=X,name='conv_3',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.conv2d(inputs=X1,name='conv_4',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.batch_normalization(inputs=X1,name='bn_2',axis=-1,center=True,scale=True)
    X1 = tf.layers.max_pooling2d(inputs=X1,name='maxpool_2',pool_size=2,strides=2,padding='same')

    X = tf.layers.conv2d(inputs=X,name='conv_4',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_5',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_3',axis=-1,center=True,scale=True)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_3',pool_size=2,strides=2,padding='same')

    X = tf.add(X,X1)
    X = tf.nn.relu(inputs=X)

    X1 = tf.layers.conv2d(inputs=X,name='conv_6',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.conv2d(inputs=X1,name='conv_7',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.batch_normalization(inputs=X1,name='bn_4',axis=-1,center=True,scale=True)
    X1 = tf.layers.max_pooling2d(inputs=X1,name='maxpool_4',pool_size=2,strides=2,padding='same')

    X = tf.layers.conv2d(inputs=X,name='conv_8',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_9',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_5',axis=-1,center=True,scale=True)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_5',pool_size=2,strides=2,padding='same')

    X = tf.add(X,X1)
    X = tf.nn.relu(inputs=X)

    X = tf.layers.flatten(inputs=X)
    X = tf.layers.dense(inputs=X,name='fc_1',units=64,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_8',axis=-1,center=True,scale=True)
    X = tf.layers.dense(inputs=X,name='fc_2',units=32,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_7',axis=-1,center=True,scale=True)
    X = tf.layers.dense(inputs=X,name='fc_3',units=16,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_8',axis=-1,center=True,scale=True)

    prLogits = tf.layers.dense(inputs=X,name='logits',units=10,activation=tf.nn.relu)
    prSoftMax = tf.nn.softmax(prLogits,name='softmax')

    return prLogits, prSoftMax

def CIFAR10Model_ResNext(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    X = Img

    X = tf.layers.conv2d(inputs=X,name='conv_1',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_2',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_1',axis=-1,center=True,scale=True)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_1',pool_size=2,strides=2,padding='same')

    X1 = tf.layers.conv2d(inputs=X,name='conv_3',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.conv2d(inputs=X1,name='conv_4',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.batch_normalization(inputs=X1,name='bn_2',axis=-1,center=True,scale=True)
    X1 = tf.layers.max_pooling2d(inputs=X1,name='maxpool_2',pool_size=2,strides=2,padding='same')

    X2 = tf.layers.conv2d(inputs=X,name='conv_4',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X2 = tf.layers.conv2d(inputs=X2,name='conv_5',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X2 = tf.layers.batch_normalization(inputs=X2,name='bn_3',axis=-1,center=True,scale=True)
    X2 = tf.layers.max_pooling2d(inputs=X2,name='maxpool_3',pool_size=2,strides=2,padding='same')

    X = tf.add(X1,X2)
    X = tf.nn.relu(inputs=X)

    X1 = tf.layers.conv2d(inputs=X,name='conv_6',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.conv2d(inputs=X1,name='conv_7',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.batch_normalization(inputs=X1,name='bn_4',axis=-1,center=True,scale=True)
    X1 = tf.layers.max_pooling2d(inputs=X1,name='maxpool_4',pool_size=2,strides=2,padding='same')

    X2 = tf.layers.conv2d(inputs=X,name='conv_8',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X2 = tf.layers.conv2d(inputs=X2,name='conv_9',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X2 = tf.layers.batch_normalization(inputs=X2,name='bn_5',axis=-1,center=True,scale=True)
    X2 = tf.layers.max_pooling2d(inputs=X2,name='maxpool_5',pool_size=2,strides=2,padding='same')

    X = tf.add(X1,X2)
    X = tf.nn.relu(inputs=X)

    X = tf.layers.flatten(inputs=X)
    X = tf.layers.dense(inputs=X,name='fc_1',units=64,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_6',axis=-1,center=True,scale=True)
    X = tf.layers.dense(inputs=X,name='fc_2',units=32,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_7',axis=-1,center=True,scale=True)
    X = tf.layers.dense(inputs=X,name='fc_3',units=16,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_8',axis=-1,center=True,scale=True)

    prLogits = tf.layers.dense(inputs=X,name='logits',units=10,activation=tf.nn.relu)
    prSoftMax = tf.nn.softmax(prLogits,name='softmax')

    return prLogits, prSoftMax

def CIFAR10Model_DenseNet(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################

    X = Img

    X = tf.layers.conv2d(inputs=X,name='conv_1',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.conv2d(inputs=X,name='conv_2',filters=32,kernel_size=3,padding='same',activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_1',axis=-1,center=True,scale=True)
    X = tf.layers.max_pooling2d(inputs=X,name='maxpool_1',pool_size=2,strides=2,padding='same')

    concat_layers_1 = []
    X1 = tf.layers.conv2d(inputs=X,name='conv_3',filters=16,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.batch_normalization(inputs=X1,name='bn_2',axis=-1,center=True,scale=True)
    concat_layers_1.append(X1)
    X2 = tf.layers.conv2d(inputs=X,name='conv_4',filters=16,kernel_size=3,padding='same',activation=tf.nn.relu)
    X2 = tf.layers.batch_normalization(inputs=X2,name='bn_3',axis=-1,center=True,scale=True)
    concat_layers_1.append(X2)
    X3 = tf.layers.conv2d(inputs=X,name='conv_5',filters=16,kernel_size=3,padding='same',activation=tf.nn.relu)
    X3 = tf.layers.batch_normalization(inputs=X3,name='bn_4',axis=-1,center=True,scale=True)
    concat_layers_1.append(X3)
    X4 = tf.layers.conv2d(inputs=X,name='conv_6',filters=16,kernel_size=3,padding='same',activation=tf.nn.relu)
    X4 = tf.layers.batch_normalization(inputs=X4,name='bn_5',axis=-1,center=True,scale=True)
    concat_layers_1.append(X4)
    X = tf.concatenate(concat_layers_1axis=-1)
    X = tf.layers.max_pooling2d(inputs=X1,name='maxpool_2',pool_size=2,strides=2,padding='same')

    concat_layers_2 = []
    X1 = tf.layers.conv2d(inputs=X,name='conv_3',filters=16,kernel_size=3,padding='same',activation=tf.nn.relu)
    X1 = tf.layers.batch_normalization(inputs=X1,name='bn_6',axis=-1,center=True,scale=True)
    concat_layers_2.append(X1)
    X2 = tf.layers.conv2d(inputs=X,name='conv_4',filters=16,kernel_size=3,padding='same',activation=tf.nn.relu)
    X2 = tf.layers.batch_normalization(inputs=X2,name='bn_7',axis=-1,center=True,scale=True)
    concat_layers_2.append(X2)
    X = tf.concatenate(concat_layers_2,axis=-1)
    X = tf.layers.max_pooling2d(inputs=X1,name='maxpool_3',pool_size=2,strides=2,padding='same')

    X = tf.layers.flatten(inputs=X)
    X = tf.layers.dense(inputs=X,name='fc_1',units=64,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_8',axis=-1,center=True,scale=True)
    X = tf.layers.dense(inputs=X,name='fc_2',units=32,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_9',axis=-1,center=True,scale=True)
    X = tf.layers.dense(inputs=X,name='fc_3',units=16,activation=tf.nn.relu)
    X = tf.layers.batch_normalization(inputs=X,name='bn_10',axis=-1,center=True,scale=True)

    prLogits = tf.layers.dense(inputs=X,name='logits',units=10,activation=tf.nn.relu)
    prSoftMax = tf.nn.softmax(prLogits,name='softmax')

    return prLogits, prSoftMax

