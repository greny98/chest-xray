import os.path

import numpy as np
import tensorflow as tf
import albumentations as augment
from tensorflow.keras.applications import densenet
import pandas as pd
from siim.configs import IMAGE_CLS_SIZE as IMAGE_SIZE, LABELS
from skmultilearn.model_selection import iterative_train_test_split

autotune = tf.data.AUTOTUNE
BATCH_SIZE = 12


def split_data(filename: str):
    info = pd.read_csv(filename)
    labels = info[LABELS]
    images = info[['image']]
    x_train, y_train, x_val, y_val = iterative_train_test_split(images.values, np.int32(labels.values),
                                                                test_size=0.075)
    return (x_train.reshape(-1), y_train), (x_val.reshape(-1), y_val)


def classify_augmentation(training=False):
    if training:
        transform = augment.Compose([
            augment.ImageCompression(quality_lower=90, quality_upper=100),
            augment.HorizontalFlip(),
            augment.VerticalFlip(),
            augment.RandomRotate90(),
            augment.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            augment.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
            augment.GaussNoise(p=0.3),
            augment.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ])
    else:
        transform = augment.Compose([augment.Resize(IMAGE_SIZE, IMAGE_SIZE)])

    def preprocess_image(image_file):
        image_raw = tf.io.read_file(image_file)
        decoded = tf.image.decode_jpeg(image_raw, channels=3)
        data = {'image': decoded.numpy()}
        aug_img = transform(**data)['image']
        aug_img = tf.cast(aug_img, tf.float32)
        tensor = densenet.preprocess_input(aug_img)
        return tensor

    return preprocess_image


def SiimClassificationGenerator(data, image_dir, batch_size=BATCH_SIZE, training=False):
    def process_data(image_file, label):
        aug_img = tf.numpy_function(func=classify_augmentation(training), inp=[image_file], Tout=tf.float32)
        return aug_img, tf.cast(label, tf.float32)

    images, y = data
    images = [os.path.join(image_dir, image) for image in images]
    images_ts = tf.data.Dataset.from_tensor_slices(images)
    labels_ts = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip((images_ts, labels_ts))
    ds = ds.shuffle(24 * batch_size, reshuffle_each_iteration=training)
    ds = ds.map(lambda x, y: process_data(x, y), num_parallel_calls=autotune).batch(batch_size)
    ds = ds.prefetch(autotune)
    return ds
