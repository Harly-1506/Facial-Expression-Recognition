import numpy as np 
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import random
from imgaug import augmenters as iaa

from albumentations import ( 

    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, 

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, 

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, 

    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ToFloat,JpegCompression, 

    Resize, Normalize, Rotate, RandomCrop, Crop, CenterCrop, DualTransform, PadIfNeeded, RandomCrop, ToGray, 

    IAAFliplr, IAASuperpixels, VerticalFlip, RandomGamma, ElasticTransform, ImageOnlyTransform ,RandomBrightnessContrast 
) 

 

# from prlab.data_augment.albumentations import ( 

#     RandomResizedCrop, HorizontalShear, MaskThreshold, BrightnessShift, BrightnessMultiply, DoGama, ShiftScaleRotateHeng, ElasticTransformHeng 

# ) 

 

def normal1_postprocessing(image): 

    image = ((image + 1) / 2.0) * 255.0 

    return image.astype(np.uint8) 

# normal_preprocessing 

 

def normal_postprocessing(image): 

    image = image * 255. 

    return image.astype(np.uint8) 

# normal_postprocessing 


seg = iaa.Sequential(
    [
        # iaa.Fliplr(p=0.5, deterministic=True),
        iaa.Fliplr(p=0.5),
        # iaa.Affine(rotate=(-30, 30), deterministic=True),
        iaa.Affine(rotate=(-30, 30)),
        # iaa.GaussianBlur(sigma=(0., 4.0), deterministic=True),
        # iaa.Dropout((0., 0.15), deterministic=True),
        # iaa.Add((-25, 25), deterministic=True),
        # iaa.CropAndPad(percent=(-0.05, 0.1), pad_cval=(0, 255), deterministic=True)
    ]
)

transforms_train = Compose([
            Rotate(limit=40),
            RandomBrightness(limit=0.1),
            # JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
            # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomRotate90(),
            # HorizontalFlip(),
            # CenterCrop(30, 30), 
            # RandomCrop(width=10, height=10),
            # RandomGamma(gamma_limit=(80, 120),always_apply=False, p=0.5),
            Transpose(),
            # Flip(),
            # CenterCrop(always_apply=False, p=0.2, height=15, width=15),
            # OpticalDistortion(),
            # GridDistortion(),
            # HueSaturationValue(),
            # Equalize(always_apply=False, p=1.0, mode='cv', by_channels=True)
            # Normalize(0,1)
            # Blur(blur_limit=3),
            
        ])

transforms_val = Compose([
            Rotate(limit=40),
            RandomBrightness(limit=0.1),

        ])


def train_fn(image):
    data = {"image":image}
    aug_data = transforms_train(**data)
    aug_img = aug_data["image"]
    # aug_img = tf.cast(aug_img, tf.float32)
    # aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
    return aug_img

def val_fn(image):
    data = {"image":image}
    aug_data = transforms_val(**data)
    aug_img = aug_data["image"]
    # aug_img = tf.cast(aug_img, tf.float32)
    return aug_img


def train_aug(image_size, p=1.0): 

    return Compose([ 

        Resize(image_size + 36, image_size + 36), 

        CenterCrop(image_size, image_size), 

        # RandomCrop(image_size, image_size, p=1), 

        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45), 

        # Resize(image_size+36, image_size+36), 

    # CenterCrop(image_size, image_size), 

        # Rotate(limit=15), 

        HorizontalFlip(p=0.5), 

        PadIfNeeded(min_height=image_size, min_width=image_size, p=1) 

    ], p=p) 

# train_aug 
    

def valid_aug(image_size, p=1.0): 

    return Compose([ 

        Resize(image_size, image_size, p=1), 

        PadIfNeeded(min_height=image_size, min_width=image_size, p=1) 

    ], p=p) 

# valid_aug 
def val_augmentation(val_ds):
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift,
        # brightness_range=True,
        vertical_flip=True,
        shear_range=0.2, 
        zoom_range=0.2,
        )
    datagen.fit(val_ds)
    return datagen 
 # data augumentation keras
def keras_augmentation(x_train, labels, num_class = 1):
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift,
        # brightness_range=True,
        vertical_flip=True,
        shear_range=0.2, 
        zoom_range=0.2,
        )
    X_fit = []
    for i in range(len(labels)):
        if labels[i] == num_class:
            X_fit.append(x_train[i])
    X_fit = np.array(X_fit)
    datagen.fit(X_fit)
    return datagen 


# offline data augmentations
def offline_augment(X_train, y_train, datagen, num_class = 0, number_aug = 0):
  augmented_data = []
  augmented_labels = []
  for i in range(len(y_train)):
    if y_train[i]==num_class:
      num_augmented = 0
      for X_batch in datagen.flow(np.expand_dims(X_train[i], axis = 0), shuffle=True):
          augmented_data.append(X_batch.astype("float64"))
          augmented_labels.append(num_class)
          num_augmented += 1
          if num_augmented == number_aug :
              break
  augmented_data = np.concatenate(augmented_data)

  X_train = np.concatenate((X_train, augmented_data), axis = 0, dtype = np.dtype("float64"))
  y_train = np.concatenate((y_train, augmented_labels), axis = 0, dtype = np.dtype("float64"))
  return X_train, y_train

def aug_offline(X_train, y_train, num_class = 1, number_aug = 0):
    datagen = keras_augmentation(X_train, y_train, num_class)
    X_new, y_new = offline_augment(X_train, y_train, datagen, num_class, number_aug)

    return X_new, y_new


 

vggface2_mean = (91.4953, 103.8827, 131.0912) # BGR 

def vggface2_preprocessing_input(x): 

    x[..., 0] -= vggface2_mean[0] 

    x[..., 1] -= vggface2_mean[1] 

    x[..., 2] -= vggface2_mean[2] 

    return x 

# vggface2_preprocessing_input 

 

def vggface2_postprocessing_input(x): 

    x[..., 0] += vggface2_mean[0] 

    x[..., 1] += vggface2_mean[1] 

    x[..., 2] += vggface2_mean[2] 

    return x 

# vggface2_postprocessing_input 

 

vggface1_mean = (91.4953, 103.8827, 131.0912) 

def vggface1_preprocessing_input(x): 

    x[..., 0] -= vggface1_mean[0] 

    x[..., 1] -= vggface1_mean[1] 

    x[..., 2] -= vggface1_mean[2] 

    return x 

# preprocessing_input_bgr 

 




def nasnet_preprocess_input(x): 

    """ 

    Image RGB 

    tf: will scale pixels between -1 and 1, sample-wise. 

    """ 

    x = x[...,::-1] # BGR --> RGB 

    x = x / 127.5 

    x -= 1.0 

    return x 

# nasnet_preprocess_input 

 

def inceptionresnetv2_preprocess_input(x): 

    """ 

    Image RGB 

    tf: will scale pixels between -1 and 1, sample-wise. 

    """ 

    x = x[...,::-1] # BGR --> RGB 

    x = x / 127.5 

    x -= 1.0 

    return x 

# inceptionresnetv2_preprocess_input 

 

def xception_preprocess_input(x): 

    """ 

    Image RGB 

    tf: will scale pixels between -1 and 1, sample-wise. 

    """ 

    x = x[..., ::-1]  # BGR --> RGB 

    x = x / 127.5 

    x -= 1.0 

    return x 

# xception_preprocess_input 

 

densenet_mean = [0.485, 0.456, 0.406] 

densenet_std = [0.229, 0.224, 0.225] 

def densenet_preprocess_input(x): 

    """ 

    torch: will scale pixels between 0 and 1 

    and then will normalize each channel with respect to the 

    ImageNet dataset. 

    """ 

    x = x[...,::-1] # BGR --> RGB 

    x /= 255. 

    x[..., 0] -= densenet_mean[0] 

    x[..., 1] -= densenet_mean[1] 

    x[..., 2] -= densenet_mean[2] 

 

    x[..., 0] /= densenet_std[0] 

    x[..., 1] /= densenet_std[1] 

    x[..., 2] /= densenet_std[2] 

    return x 

# densenet_preprocess_input 

 

def rafdb_train_aug(crop_size, image_size, p=1.0): 

    return Compose([ 

        Resize(crop_size[0], crop_size[1], p=1), 

        RandomCrop(image_size[0], image_size[1], p = 1), 

        Rotate(limit = 45), 

        # ToFloat(max_value  = 255.0), 

        OneOf([ 

            HorizontalFlip(p=0.5), 

            OneOf([ 

                RandomResizedCrop(limit=0.125), 

                # HorizontalShear(origin_img_size, np.random.uniform(-0.07,0.07)), 

                ShiftScaleRotateHeng(dx=0, dy=0, scale=1, angle=np.random.uniform(0, 10)), 

                ElasticTransformHeng(grid=10, distort=np.random.uniform(0, 0.15)), 

            ]), 

            OneOf([ 

                BrightnessShift(np.random.uniform(-0.05, 0.05)), 

                BrightnessMultiply(np.random.uniform(1-0.05, 1+0.05)), 

                DoGama(np.random.uniform(1-0.05, 1+0.05)), 

            ]) 

        ]), 

        PadIfNeeded(min_height=image_size[0], min_width=image_size[1], p=1), 

#        Normalize(mean=0.5, std=0.5, max_pixel_value = 1.0), 

    ], p=p) 

# # tumor_preprocess 