# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W6VigdfBIo94ZuJb3qVqzyGV1iC5_kV4
"""

import keras
import tensorflow as tf
import cv2 

import matplotlib.pyplot as plt 
import numpy as np 

from keras.preprocessing.image import ImageDataGenerator

from keras import layers, Input

from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation,Average
from tensorflow.keras.applications import VGG16, ResNet50, VGG19, ResNet50V2, MobileNetV2, MobileNetV2
from tensorflow.keras.optimizers import Adam, Adamax
from keras.models import load_model, Model
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def Pre_train_model(Pre_train, input_tensor , input_shape = None):

  baseModel = Pre_train(weights="imagenet", include_top= False, input_tensor = input_tensor, input_shape = input_shape)
  inputs = baseModel.layers[0].input
  output = baseModel.output
  fcHead = layers.GlobalAveragePooling2D()(output)
  fcHead = layers.Dense(1024,  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.L2(1e-4),
                        activity_regularizer=regularizers.L2(1e-5))(fcHead)
  fcHead = layers.Activation('relu')(fcHead)
  fcHead = layers.Dropout(0.5)(fcHead)

  fcHead = layers.Dense(256,  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.L2(1e-4),
                        activity_regularizer=regularizers.L2(1e-5))(fcHead)
  fcHead = layers.Activation('relu')(fcHead)
  fcHead = layers.Dropout(0.5)(fcHead)

  fcHead = layers.Dense(128,  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.L2(1e-4),
                        activity_regularizer=regularizers.L2(1e-5))(fcHead)
  fcHead = layers.Activation('relu')(fcHead)
  fcHead = layers.Dropout(0.5)(fcHead)

  fcHead = layers.Dense(7)(fcHead)
  fcHead = layers.Activation('Softmax')(fcHead)
  model = keras.Model(inputs=inputs, outputs=fcHead)
  
  return model
#  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
#                         bias_regularizer=regularizers.L2(1e-4),
#                         activity_regularizer=regularizers.L2(1e-5)

def Pre_train_model1(Pre_train, input_tensor , input_shape = None):

  baseModel = Pre_train(weights="imagenet", include_top= False, input_tensor = input_tensor, input_shape = None)
  inputs = baseModel.layers[0].input
  output = baseModel.output
  fcHead = layers.GlobalAveragePooling2D()(output)
  fcHead = layers.Dense(1024,  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.L2(1e-4),
                        activity_regularizer=regularizers.L2(1e-5))(fcHead)
  fcHead = layers.Activation('relu')(fcHead)
  fcHead = layers.Dropout(0.2)(fcHead)

  fcHead = layers.Dense(256,  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.L2(1e-4),
                        activity_regularizer=regularizers.L2(1e-5))(fcHead)
  fcHead = layers.Activation('relu')(fcHead)
  fcHead = layers.Dropout(0.2)(fcHead)

  fcHead = layers.Dense(128,  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.L2(1e-4),
                        activity_regularizer=regularizers.L2(1e-5))(fcHead)
  fcHead = layers.Activation('relu')(fcHead)
  fcHead = layers.Dropout(0.2)(fcHead)

  fcHead = layers.Dense(64,  kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.L2(1e-4),
                        activity_regularizer=regularizers.L2(1e-5))(fcHead)
  fcHead = layers.Activation('relu')(fcHead)
  fcHead = layers.Dropout(0.2)(fcHead)
  fcHead = layers.Dense(7, activation='softmax')(fcHead)
  model = keras.Model(inputs=inputs, outputs=fcHead)
  
  return model
def Confusion_Matrix(model, x_test, y_test,  emotion , mode = None): 
  
  y_pred = model.predict(x_test)
  y_pred = np.argmax(y_pred ,axis = 1)
  y_true=np.argmax(y_test,axis = 1)
  a = precision_recall_fscore_support(y_true, y_pred, average='macro')
  cf = confusion_matrix(y_true,y_pred)

  if mode is True:
    cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]

  print(f'+ precision = {a[0]:.3f}')
  print(f'+ recall = {a[1]:.3f}')
  print(f'+ f1_score = {a[2]:.3f}')
  plt.figure(figsize = (10,8))
  if mode is True:
    ax = sns.heatmap(cf, annot=True, 
            fmt='.2%', cmap='Blues')
  else:
    ax = sns.heatmap(cf,fmt="d",annot = True, 
             cmap='Blues')
  ax.xaxis.set_ticklabels(emotion)
  ax.yaxis.set_ticklabels(emotion)
  plt.xlabel("True_labels")
  plt.ylabel("Predicted labels")
  plt.title("Confusion Matrix")
  plt.show(ax)

def plot_loss(H, numOfEpochs):
  
  fig = plt.figure()

  plt.plot(np.arange(0, numOfEpochs), H.history['loss'], label='training loss')
  plt.plot(np.arange(0, numOfEpochs), H.history['val_loss'], label='validation loss')
  
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

def plot_acc(H, numOfEpochs):

  fig = plt.figure()

  plt.plot(np.arange(0, numOfEpochs), H.history['accuracy'], label='training accuracy')
  plt.plot(np.arange(0, numOfEpochs), H.history['val_accuracy'], label='validation accuracy')
  
  plt.title('Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

def save_history(path_json, history):
  hist_df = pd.DataFrame(history.history) 

  # save to json:  
  with open(path_json, mode='w') as f:
      hist_df.to_json(f)
# def save_model(model,path)

'''
Attention on Pretrained Model

'''
