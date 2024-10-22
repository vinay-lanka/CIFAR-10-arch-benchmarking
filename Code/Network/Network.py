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

def BaseModel(Img, ImageSize, MiniBatchSize):
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
    layer = tf.layers.conv2d(inputs=Img, name='conv_layer_1', filters = 32, kernel_size = 3, activation = tf.nn.relu)
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=2, strides=2)
    layer = tf.layers.conv2d(inputs=layer, name='conv_layer_2', filters = 64, kernel_size = 3, activation = tf.nn.relu)
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=2, strides=2)
    
    layer = tf.layers.flatten(layer)

    layer = tf.layers.dense(layer, name='fully_connected_layer_1', units = 128, activation = tf.nn.relu)
    prLogits = tf.layers.dense(layer, name='fully_connected_layer_2', units = 10, activation = None)

    prSoftMax = tf.nn.softmax(logits = prLogits)  
   

    return prLogits, prSoftMax

def BaseModelModified(Img, ImageSize, MiniBatchSize):

    layer = tf.layers.conv2d(inputs=Img, name='conv_layer_1', filters = 32, kernel_size = 3, activation = tf.nn.relu)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.conv2d(inputs=layer, name='conv_layer_2', filters = 32, kernel_size = 3, activation = tf.nn.relu)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=2, strides=2)

    layer = tf.layers.conv2d(inputs=layer, name='conv_layer_3', filters = 64, kernel_size = 3, activation = tf.nn.relu)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.conv2d(inputs=layer, name='conv_layer_4', filters = 64, kernel_size = 3, activation = tf.nn.relu)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=2, strides=2)

    # layer = tf.layers.conv2d(inputs=layer, name='conv_layer_5', filters = 128, kernel_size = 3, activation = tf.nn.relu)
    # layer = tf.layers.batch_normalization(layer)
    # layer = tf.layers.conv2d(inputs=layer, name='conv_layer_6', filters = 128, kernel_size = 3, activation = tf.nn.relu)
    # layer = tf.layers.batch_normalization(layer)
    # layer = tf.layers.max_pooling2d(inputs=layer, pool_size=2, strides=2)
    
    layer = tf.layers.flatten(layer)
    layer = tf.layers.dropout(layer,rate=0.2)

    layer = tf.layers.dense(layer, name='fully_connected_layer_1', units = 256, activation = tf.nn.relu)
    layer = tf.layers.dropout(layer,rate=0.2)

    layer = tf.layers.dense(layer, name='fully_connected_layer_2', units = 128, activation = tf.nn.relu)
    layer = tf.layers.dropout(layer,rate=0.2)
    
    prLogits = tf.layers.dense(layer, name='fully_connected_layer_3', units = 10, activation = None)

    prSoftMax = tf.nn.softmax(logits = prLogits)  
   

    return prLogits, prSoftMax

def conv_residual_block(Img, filters, kernel_size, block):
    skip = tf.layers.conv2d(inputs=Img, name='conv_layer_skip_block_'+str(block), padding='same', filters = filters, kernel_size = kernel_size, activation = None)
    skip = tf.layers.batch_normalization(skip)

    layer = tf.layers.conv2d(inputs=Img, name='conv_layer_1_block_'+str(block), padding='same', filters = filters, kernel_size = kernel_size, activation = None)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.nn.relu(layer, name='relu_1_block_' + str(block))

    layer = tf.layers.conv2d(inputs=layer, name='conv_layer_2_block_'+str(block), padding='same', filters = filters, kernel_size = kernel_size, activation = None)
    layer = tf.layers.batch_normalization(layer)
    layer = tf.nn.relu(layer, name='relu_2_block_' + str(block))

    layer = tf.layers.conv2d(inputs=layer, name='conv_layer_3_block_'+str(block), padding='same', filters = filters, kernel_size = kernel_size, activation = None)
    layer = tf.layers.batch_normalization(layer)

    output = tf.math.add(layer, skip)
    output = tf.nn.relu(output, name='final_relu')
    return output

def ResNetModel(Img, ImageSize, MiniBatchSize):

    resblock = conv_residual_block(Img, filters=32, kernel_size=3, block=1)
    resblock = conv_residual_block(resblock, filters=32, kernel_size=3, block=2)
    resblock = conv_residual_block(resblock, filters=32, kernel_size=3, block=3)

    layer = tf.layers.flatten(resblock)
    layer = tf.layers.dense(layer, name='fc1', units = 256, activation = tf.nn.relu)
    layer = tf.layers.dense(layer, name='fc2', units = 128, activation = tf.nn.relu)
    prLogits = tf.layers.dense(layer, name='fc3', units = 10, activation = None)

    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax

def conv_resnext_block(Img, filters, kernel_size, block, cardinality):
    skip = tf.layers.conv2d(inputs=Img, name='conv_layer_skip_block_'+str(block), padding='same', filters = filters, kernel_size = kernel_size, activation = None)
    skip = tf.layers.batch_normalization(skip)

    main = tf.layers.conv2d(inputs=Img, name='main_'+'conv_layer_1_block_'+str(block), padding='same', filters = 32, kernel_size = kernel_size, activation = None)
    main = tf.layers.batch_normalization(main)
    main = tf.nn.relu(main, name='main_'+'relu_1_block_' + str(block))

    for i in range(cardinality):
        layer = tf.layers.conv2d(inputs=main, name=str(i)+'conv_layer_1_block_'+str(block), padding='same', filters = 4, kernel_size = kernel_size, activation = None)
        layer = tf.layers.batch_normalization(layer)
        layer = tf.nn.relu(layer, name=str(i)+'relu_1_block_' + str(block))

        layer = tf.layers.conv2d(inputs=layer, name=str(i)+'conv_layer_2_block_'+str(block), padding='same', filters = 4, kernel_size = kernel_size, activation = None)
        layer = tf.layers.batch_normalization(layer)
        layer = tf.nn.relu(layer, name=str(i)+'relu_2_block_' + str(block))

        layer = tf.layers.conv2d(inputs=layer, name=str(i)+'conv_layer_3_block_'+str(block), padding='same', filters = 32, kernel_size = kernel_size, activation = None)
        layer = tf.layers.batch_normalization(layer)
        main = tf.math.add(main, layer)

    
    output = tf.math.add(main, skip)
    output = tf.nn.relu(output, name='final_relu')
    return output

def ResNeXtModel(Img, ImageSize, MiniBatchSize):

    resblock = conv_resnext_block(Img, filters=32, kernel_size=3, block=1, cardinality=8)
    resblock = conv_resnext_block(resblock, filters=32, kernel_size=5, block=2, cardinality=8)
    # resblock = conv_residual_block(resblock, filters=128, kernel_size=5, block=3)

    layer = tf.layers.flatten(resblock)
    layer = tf.layers.dense(layer, name='fc1', units = 256, activation = tf.nn.relu)
    # layer = tf.layers.dense(layer, name='fc2', units = 128, activation = tf.nn.relu)
    prLogits = tf.layers.dense(layer, name='fc3', units = 10, activation = None)

    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax

def inception_block(Img, filters, block):
    conv_one = tf.layers.conv2d(inputs=Img, name='conv_one'+str(block), padding='same', filters = filters, kernel_size = 1, activation = None)
    # conv_one = tf.layers.batch_normalization(conv_one)
    conv_one = tf.nn.relu(conv_one, name='conv_one_relu_block_' + str(block))
     
    conv_three = tf.layers.conv2d(inputs=conv_one, name='conv_three'+str(block), padding='same', filters = filters, kernel_size = 3, activation = None)
    # conv_three = tf.layers.batch_normalization(conv_three)
    conv_three = tf.nn.relu(conv_three, name='conv_three_relu_block_' + str(block))

    conv_five = tf.layers.conv2d(inputs=conv_one, name='conv_five'+str(block), padding='same', filters = filters, kernel_size = 5, activation = None)
    # conv_five = tf.layers.batch_normalization(conv_five)
    conv_five = tf.nn.relu(conv_five, name='conv_five_relu_block_' + str(block))

    max_pool_three = tf.layers.max_pooling2d(inputs=Img, pool_size=3, strides=1, padding='same')
    conv_one_after_pool = tf.layers.conv2d(inputs=max_pool_three, name='conv_one_after_pool_'+str(block), padding='same', filters = filters, kernel_size = 1, activation = None)
    # conv_one_after_pool = tf.layers.batch_normalization(conv_one_after_pool)
    conv_one_after_pool = tf.nn.relu(conv_one_after_pool, name='conv_one_after_pool_relu_block_' + str(block))

    output = tf.concat([conv_one, conv_three, conv_five, conv_one_after_pool], axis=-1)

    return output

def transition_layer(layer, filters, block):
    conv_one = tf.layers.conv2d(inputs=layer, name='conv_one_transition_'+str(block), padding='same', filters = filters, kernel_size = 1, activation = None)
    max_pool_two = tf.layers.max_pooling2d(inputs=conv_one, pool_size=2, strides=2, padding='same')
    return max_pool_two

def DenseNetModel(Img, ImageSize, MiniBatchSize):

    incblock = inception_block(Img, filters=32, block=1)
    transfer_block = transition_layer(incblock, filters=32, block=2)
    incblock = inception_block(transfer_block, filters=32, block=3)
    transfer_block = transition_layer(incblock, filters=32, block=4)
    # resblock = conv_residual_block(resblock, filters=128, kernel_size=5, block=3)

    layer = tf.layers.flatten(transfer_block)
    layer = tf.layers.dense(layer, name='fc1', units = 256, activation = tf.nn.relu)
    # layer = tf.layers.dense(layer, name='fc2', units = 128, activation = tf.nn.relu)
    prLogits = tf.layers.dense(layer, name='fc3', units = 10, activation = None)

    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax