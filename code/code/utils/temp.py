from IPython import display 

import numpy as np 

import pandas as pd 

import tensorflow as tf 

import data_augmentation

import data_processing



def tf_parse_data (dataset, in_dtype=tf.float32, out_dtype=tf.float32): 

    n_inputs = 1

    n_outputs = 1

    T_out = [in_dtype] * n_inputs + [out_dtype] * n_outputs 

 

    def internal (idx): 

        idx = idx.numpy() 

        x, y = dataset[idx] 

        return [*list(x), *list(y)] 

 

    def tf_internal (idx): 

        result = tf.py_function(internal, [idx], T_out) 

        return tuple(result[:n_inputs]), tuple(result[n_inputs:]) 

 

    # tf_internal 

    return tf_internal 

# tf_parse_data 

 


class DataGenerator(): 

    def __init__(self, list_images, 

                 list_labels, # ["train", "val", "test"] 

                 shape = (48,48,1),

                 n_classes = 7,

                 shuffle = True,

                 augmentation = None

                 ): 

        self.list_images = list_images  

        self.list_labels = list_labels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.img_indexes = np.arange(len(self.list_images))

        self.shape = shape

        self.augmentation = augmentation


    def __len__ (self): 

      return (len(self.img_indexes))

    # __len__ 

 
    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      i = self.img_indexes[index]

      x , y = (), ()

      image = self.list_images[i]
      image = image.reshape(self.shape)
      image = image/255.0
      if self.augmentation is True:
          
        image = data_augmentation.aug_fn(image)
        # for pos , img in enumerate(image):
        
        x += (image,)
      else:
           x += (image,)
      # x.append(img)
      labels = tf.keras.utils.to_categorical(self.list_labels[i], num_classes=self.n_classes)
      y += (labels,)
      # y.append(labels)

      return x, y

      pass
 

 

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

            ds = ds.shuffle(buffer_size=shuffle_size, seed=seed, reshuffle_each_iteration=True) # .repeat()
        if batch_size>0: 

            ds = ds.batch(batch_size=batch_size) 

        ds = ds.prefetch(prefetch_size) 

        return ds 

