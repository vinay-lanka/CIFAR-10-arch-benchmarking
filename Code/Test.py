#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import *
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
import math as m
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import pickle


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')

    return ImageSize, DataPath
    
def ReadImages(Model, ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = cv2.imread(ImageName).astype('float32')
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
        

    #Data Augmentation
    # I1 = tf.image.random_flip_left_right(I1)
    # I1 = tf.image.random_hue(I1,0.2)
    # plt.imshow(I1)
    # plt.show()
    if Model != 'BaseModel':
        mean = np.mean(I1, axis=(0,1,2), keepdims=True)
        std = np.std(I1, axis=(0,1,2), keepdims=True)
        # print("mean",mean)
        # print("std", std)
        standardized_image = (I1 - mean) / (std) 
        I1Combined = np.expand_dims(standardized_image, axis=0)
    else:
        I1Combined = np.expand_dims(I1, axis=0)
    
    return I1Combined, I1
                

def TestOperation(Model, ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    if Model == 'BaseModel':
        _, prSoftMaxS = BaseModel(ImgPH, ImageSize, 1)
    elif Model == "BaseModelModified":
        _, prSoftMaxS = BaseModelModified(ImgPH, ImageSize, 1)
    elif Model == "ResNetModel":
        _, prSoftMaxS = ResNetModel(ImgPH, ImageSize, 1)
    elif Model == "ResNeXtModel":
        _, prSoftMaxS = ResNeXtModel(ImgPH, ImageSize, 1)
    elif Model == "DenseNetModel":
        _, prSoftMaxS = DenseNetModel(ImgPH, ImageSize, 1)
    # Setup Saver
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        OutSaveT = open(LabelsPathPred, 'w')

        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            Img, ImgOrg = ReadImages(Model, ImageSize, DataPathNow)
            FeedDict = {ImgPH: Img}
            PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))

            OutSaveT.write(str(PredT)+'\n')
            
        OutSaveT.close()

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(Model, String, LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("./results/"+ Model + "/" + String +"ConfusionMatrix.png")
    plt.show()

    # # Print the confusion matrix as text.
    # for i in range(10):
    #     print(str(cm[i, :]) + ' ({0})'.format(i))

    # # Print the class-numbers for easy reference.
    # class_numbers = [" ({0})".format(i) for i in range(10)]
    # print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')

def TestAccuracyPerEpoch(Model, ImgPH, ImageSize, ModelPath, DataPath):
     # Predict output with forward pass, MiniBatchSize for Test is 1
    if Model == 'BaseModel':
        _, prSoftMaxS = BaseModel(ImgPH, ImageSize, 1)
    elif Model == "BaseModelModified":
        _, prSoftMaxS = BaseModelModified(ImgPH, ImageSize, 1)
    elif Model == "ResNetModel":
        _, prSoftMaxS = ResNetModel(ImgPH, ImageSize, 1)
    elif Model == "ResNeXtModel":
        _, prSoftMaxS = ResNeXtModel(ImgPH, ImageSize, 1)
    elif Model == "DenseNetModel":
        _, prSoftMaxS = DenseNetModel(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)

        LabelsPred = []

        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            Img, ImgOrg = ReadImages(Model, ImageSize, DataPathNow)
            FeedDict = {ImgPH: Img}
            PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))

            LabelsPred.append(float(PredT))
            # OutSaveT.write(str(PredT)+'\n')
            
        # OutSaveT.close()
    return LabelsPred

def ConfusionMatrixMod(LabelsTrue, LabelsPred):
    Acc = Accuracy(LabelsPred, LabelsTrue)
    print('Accuracy: '+ str(Acc), '%')
    return Acc
        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Model', default='ResNeXtModel', help='Enter model name')
    Parser.add_argument('--Epoch', default=15, help='Enter model name')
    # Parser.add_argument('--ModelPath', dest='ModelPath', default='./Checkpoints/BaseModel/9model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='./CIFAR10', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./Code/TxtFiles/', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    Epoch = Args.Epoch
    ModelPath = './Checkpoints/' + Args.Model + '/' +  str(int(Epoch)-1) + 'model.ckpt'
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    ImageSize, DataPathTrain = SetupAll(BasePath + '/Train/')
    ImageSize, DataPathTest = SetupAll(BasePath + '/Test/')


    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    LabelsPathTrain = LabelsPath + 'LabelsTrain.txt'
    LabelsPathTest = LabelsPath + 'LabelsTest.txt'
    LabelsPathPredTrain = LabelsPath + Args.Model + '/PredOutTrain.txt' # Path to save predicted labels
    LabelsPathPredTest = LabelsPath + Args.Model + '/PredOutTest.txt' # Path to save predicted labels

    TestOperation(Args.Model, ImgPH, ImageSize, ModelPath, DataPathTrain, LabelsPathPredTrain)
    tf.reset_default_graph()
    print(ModelPath)
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    TestOperation(Args.Model, ImgPH, ImageSize, ModelPath, DataPathTest, LabelsPathPredTest)

    #Get training stats from binary
    train_loss_per_epoch = []
    train_accuracy_per_epoch = []
    with open(LabelsPath + Args.Model + '/ModelTrainingStats.bin', 'rb') as StatReader:
        [train_loss_per_epoch, train_accuracy_per_epoch] = pickle.load(StatReader)

    LabelsTrueTest, LabelsPredTest = ReadLabels(LabelsPathTest, LabelsPathPredTest)
    LabelsTrueTest = list(LabelsTrueTest)

    epochs = np.arange(start=0, stop=Epoch, step=1)
    test_accuracy_per_epoch = []
    for ep in epochs:
        ModelEpochCheckpoint = './Checkpoints/' + Args.Model + '/' +  str(int(ep)) + 'model.ckpt'
        print(ModelEpochCheckpoint)
        tf.reset_default_graph()
        ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
        labels_pred_per_epoch = TestAccuracyPerEpoch(Args.Model, ImgPH, ImageSize, ModelEpochCheckpoint, DataPathTest)
        test_accuracy_per_epoch.append(ConfusionMatrixMod(LabelsTrueTest, labels_pred_per_epoch))
    # print(test_accuracy_per_epoch)

    #Plotting
    plt.subplots(2, 1, figsize=(15,15))
    
    plt.subplot(2, 1, 1) #loss
    plt.plot(epochs, train_loss_per_epoch)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xlim((0,Epoch))


    plt.subplot(2, 1, 2) #Acc
    plt.plot(epochs, train_accuracy_per_epoch)
    plt.plot(epochs, test_accuracy_per_epoch)
    plt.xlabel("epochs")
    plt.xlim((0,Epoch))
    plt.ylim((0,100))
    plt.savefig("./results/"+ Args.Model +"/loss_accuracy_plot.png")

    plt.show()

    # Plot Confusion Matrix
    LabelsTrueTrain, LabelsPredTrain = ReadLabels(LabelsPathTrain, LabelsPathPredTrain)
    LabelsTrueTrain = list(LabelsTrueTrain)
    LabelsPredTrain = list(LabelsPredTrain)
    ConfusionMatrix(Args.Model, "train",  LabelsTrueTrain, LabelsPredTrain)
    # Plot Confusion Matrix
    LabelsTrueTest, LabelsPredTest = ReadLabels(LabelsPathTest, LabelsPathPredTest)
    LabelsTrueTest = list(LabelsTrueTest)
    LabelsPredTest = list(LabelsPredTest)
    # print(LabelsPredTest)
    ConfusionMatrix(Args.Model, "test", LabelsTrueTest, LabelsPredTest)
     
if __name__ == '__main__':
    main()
 
