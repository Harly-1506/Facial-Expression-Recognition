# -*- coding: utf-8 -*-
"""get_datasets.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uxqh0x1oO--QxQOpkOMxFCz3E8ClbVL3
"""



# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/AI Research/


"""## RaF DB creation

"""

import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join
from imutils import paths
from keras.preprocessing.image import img_to_array

import csv

import matplotlib.pyplot as plt
from math import ceil
from sklearn import model_selection
from keras.utils import np_utils



CASC_PATH = "/content/haarcascade_frontalface_default.xml"
SIZE_FACE = 48
SCALEFACTOR = 1.1
# EMOTIONS = ['angry', 'disgusted', 'fearful','happy', 'sad', 'surprised', 'neutral']
#RAF_DB
RAF_EMOTIONS = ['surprised', 'fearful', 'disgusted', 'happy', 'sad', 'angry', 'neutral']
SAVE_DIRECTORY ="/content/drive/MyDrive/WorkSpace/AI_Research/datasets/RAFDB"
SAVE_IMAGES = 'images.npy'
SAVE_LABELS = 'labels.npy'
SAVE_IMAGES1 = 'images1.npy'
SAVE_LABELS1 = 'labels1.npy'

PATH_FER_PLUS = "/content/drive/MyDrive/WorkSpace/AI_Research/datasets/FerPlus/fer2013.csv"
PATH_FER_PLUS_LABELS = "/content/drive/MyDrive/WorkSpace/AI_Research/datasets/FerPlus/fer2013new.csv"
PATH_FER = "/content/drive/MyDrive/WorkSpace/AI_Research/datasets/Fer/icml_face_data.csv"
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# emotions = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear'}

cascade_classifier = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, CASC_PATH))

def Raf2Vec(x):
    d = np.zeros(len(RAF_EMOTIONS))
    d[x-1]= 1.0
    return d

def Img2Array(image_path):
    data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    array = cv2.resize(data, (SIZE_FACE, SIZE_FACE),interpolation=cv2.INTER_CUBIC)
    image = np.expand_dims(array, axis = 0)
    image = img_to_array(array)
    return image

def Get_raf_aligned(dir, txt):
  dict_train, dict_test = {}, {}

  for line in open(txt, "r"):
    line = line.strip()
    if line.startswith("train"):
      dict_train[line.split(' ')[0][:11] + '_aligned.jpg'] = int(line.split(' ')[1])
    elif line.startswith("test"):
      dict_test[line.split(' ')[0][:9] + '_aligned.jpg'] = int(line.split(' ')[1])
    else:
      print("something wrong!!!")

  images, labels = [], []
  for k , v in dict_train.items():
    image_path = dir + k
    image = Img2Array(image_path)
    images.append(image)
    labels.append(v-1)

  # images1, labels1 = [], []
  for k , v in dict_test.items():
    image_path = dir + k
    image = Img2Array(image_path)
    images.append(image)
    labels.append(v-1)

  np.save(join(SAVE_DIRECTORY, 'raf_aligned_' + SAVE_IMAGES), images)
  np.save(join(SAVE_DIRECTORY, 'raf_aligned_' + SAVE_LABELS), labels)



  # np.save(join(SAVE_DIRECTORY, 'raf_aligned_' + SAVE_IMAGES1), images1)
  # np.save(join(SAVE_DIRECTORY, 'raf_aligned_' + SAVE_LABELS1), labels1)

  return "Get aligned!!!"



"""## FerPlus Creation"""

def get_ferpus():
    data = pd.read_csv(PATH_FER_PLUS)
    labels = pd.read_csv(PATH_FER_PLUS_LABELS)

    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    n_samples = len(data)
    w = 48
    h = 48

    y = np.array(labels[orig_class_names])
    X = np.zeros((n_samples, w, h, 1))
    for i in range(n_samples):
        X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))

    return X, y


def ferplus_processing(X, y):
    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    # Using mask to remove unknown or NF images
    y_mask = y.argmax(axis=-1)
    mask = y_mask < orig_class_names.index('unknown')
    X = X[mask]
    y = y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    # Add contempt to neutral and remove it
    y[:, 0] += y[:, 7]
    y = y[:, :7]
 
    X = X
    return X, y


# def split_data(X, y):
#     test_size = ceil(len(X) * 0.1)

#     # Split Data
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)

#     return x_train, y_train, x_test, y_test

def split_data(X, y):
    test_size = ceil(len(X) * 0.1)

    # Split Data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size,
                                                                      random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test


def majority_vote(list_labels):
  new_list_labels = np.argmax(list_labels, axis = 1)

  return new_list_labels

## Fer Creation ###

def get_fer(data):
    """ Prepare data for modeling 
        input: data frame with labels und pixel data
        output: image and label array """
    
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image
        
    return image_array, image_label


def load_fer(data):
    train_image_array, train_image_label = get_fer(data[data[' Usage']=='Training'])
    val_image_array, val_image_label = get_fer(data[data[' Usage']=='PrivateTest'])
    test_image_array, test_image_label = get_fer(data[data[' Usage']=='PublicTest'])

    train_images = train_image_array.reshape((train_image_array.shape[0], 48, 48, 1))
    train_images = train_images.astype(np.uint8)
    val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1))
    val_images = val_images.astype(np.uint8)
    test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1))
    test_images = test_images.astype(np.uint8)

    return train_images, train_image_label, val_images, val_image_label, test_images, test_image_label
### SOME PLOT
def plot_compare_distributions(array1, array2,emotions, title1='', title2='' ):
    df_array1 = pd.DataFrame()
    df_array2 = pd.DataFrame()
    df_array1['emotion'] = array1.argmax(axis=1)
    df_array2['emotion'] = array2.argmax(axis=1)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 8), sharey=False)
    x = emotions.values()
    
    y = df_array1['emotion'].value_counts()
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    axs[0].bar(x, y.sort_index(), )
    axs[0].set_title(title1)
    axs[0].grid()
    
    
    y = df_array2['emotion'].value_counts()
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    axs[1].bar(x, y.sort_index())
    axs[1].set_title(title2)
    axs[1].grid()
    
    plt.show()


def plot_examples(data, labels, number_labels = 0):
    fig, axs = plt.subplots(1, 5, figsize=(25, 12))
    fig.subplots_adjust(hspace = .2, wspace=.2)
    axs = axs.ravel()
    count = []
    for i in range(len(labels)):
        if np.argmax(labels[i]) == number_labels:
            count.append(data[i])
    for i in range(5):
        axs[i].imshow(count[i][:,:,0], cmap='gray')
        # axs[i].set_title(emotions[labels[i].argmax()])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

