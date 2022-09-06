from IPython import display 

import numpy as np 

import pandas as pd 

import tensorflow as tf 

from utils.processing import data_augmentation

from utils.processing import data_processing

import random 

import cv2

import albumentations as A

from keras.applications import imagenet_utils



def tf_parse_data (dataset, in_dtype=tf.float32, out_dtype=tf.float32): 

    n_inputs = 1

    n_outputs = 1

    shape = dataset.shape

    n_classes = dataset.n_classes

    n_channels = dataset.n_channels

    T_out = [in_dtype] * n_inputs + [out_dtype] * n_outputs 

 

    def internal (idx): 

        idx = idx.numpy() 

        x, y = dataset[idx] 

        return [*list(x), *list(y)] 

 

    def tf_internal (idx): 

        result = tf.py_function(internal, [idx], T_out) 

        return result

        # tf_internal 
 

    def _fixup_shape(idx):
        
        images , labels = tf_internal(idx)

        images.set_shape([*shape, n_channels])
        
        labels.set_shape([n_classes,])
        
        return images, labels

    return _fixup_shape 

    #set shape for tensor

# tf_parse_data 

#dataset_model_version


class DataGenerator(): 

    def __init__(self, list_images, 

                 list_labels, 

                 config,

                 shuffle = True,

                 augmentation = None,

                 ): 

        self.list_images = list_images  

        self.list_labels = list_labels

        self.n_classes = config["n_classes"]

        self.n_channels = config["n_channels"]

        self.shape = (config["image_size"], config["image_size"])

        self.shuffle = shuffle

        self.img_indexes = np.arange(len(self.list_images))

        self.augmentation = augmentation






    def __len__ (self): 

      return (len(self.img_indexes))

    # __len__ 

    def __getitem__(self, index):

      #get idx for each image and labels
      i = self.img_indexes[index]
      x , y = (), ()
      image = self.list_images[i]
      img_size = self.shape[0]

      image = cv2.resize(image, self.shape)

      image = np.dstack([image] * 3)
      
      # image = np.moveaxis(image, -1, 0)
      # image.shape
      
      if self.augmentation is True:
        image = data_augmentation.seg(image=image)
        image = image/255.0
        # image = tf.keras.utils.array_to_img(image)
        # image = tf.keras.preprocessing.image.array_to_img(image,  data_format = "channels_first")
        # image = tf.convert_to_tensor(image, dtype=tf.float32)
        x += (image,)
      else:
        # image = data_augmentation.val_fn(image)
        image = image/255.0

        # image = tf.keras.utils.array_to_img(image)

        x += (image,)

     
      labels = tf.keras.utils.to_categorical(self.list_labels[i], num_classes=self.n_classes)
      y += (labels,)

      return x, y


 

 

    def to_dataset(self, tf_parse_call = tf_parse_data, tf_parse_params = {"in_dtype": tf.float32, "out_dtype": tf.float32}, 

                   data_type = "train", # train ==> shuffle, repeat, valid ==> no shuffle and repeat 

                   in_dtype=tf.float32, out_dtype=tf.float32, # mapping 

                   num_parallel_calls=tf.data.AUTOTUNE, cycle_length=4, block_length=2, # interleave 

                   batch_size=32, # batch (=0 --> no batch) 

                   prefetch_size=tf.data.AUTOTUNE, # prefetch 

                   shuffle_size=4, seed=None, # shuffle 

                   ): 

        def internal_mapping_fn(idx): 

            return tf.data.Dataset.from_tensors(idx).map(tf_parse_call(self, **tf_parse_params), num_parallel_calls=num_parallel_calls) 

 

        ds = tf.data.Dataset.range(len(self)) 

        ds = ds.interleave(internal_mapping_fn, cycle_length=cycle_length, block_length=block_length) 

        
        if data_type=="train": 

            ds = ds.shuffle(buffer_size=shuffle_size, seed=seed, reshuffle_each_iteration=True).repeat()
        if batch_size>0: 

            ds = ds.batch(batch_size=batch_size)  

        ds = ds.prefetch(prefetch_size) 

        return ds 

