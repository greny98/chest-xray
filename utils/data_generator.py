import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet_v2

from static_values.values import IMAGE_SIZE, BATCH_SIZE

autotune = tf.data.AUTOTUNE

data_augmentation_seq = tf.keras.Sequential([
    layers.RandomRotation(0.4),
    layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
    layers.RandomContrast(0.5),
    layers.RandomZoom(0.15),
])


def data_augmentation(training=False, image_size=IMAGE_SIZE):
    def preprocessing_image(image, y):
        image_raw = tf.io.read_file(image)
        decoded = tf.image.decode_jpeg(image_raw, channels=3)
        decoded = tf.cast(decoded, tf.float32)
        tensor = resnet_v2.preprocess_input(decoded)
        tensor = tf.image.resize(tensor, size=(image_size, image_size))
        if not training:
            return tensor, y
        # random augmentation
        if np.random.random_sample(1) > 0.5:
            tensor = tf.image.random_brightness(tensor, 0.5)
        return tensor, y

    return preprocessing_image


def create_ds(images, y, image_dir, image_size=IMAGE_SIZE, training=False, batch_size=BATCH_SIZE):
    images_ts = tf.data.Dataset.from_tensor_slices(image_dir + images)
    labels_ts = []
    for col in range(y.shape[1]):
        labels_ts.append(tf.data.Dataset.from_tensor_slices(y[:, col].astype(float)))
    labels_ts = tf.data.Dataset.zip(tuple(labels_ts))
    ds = tf.data.Dataset.zip((images_ts, labels_ts))
    ds = ds.shuffle(16 * batch_size, reshuffle_each_iteration=training)
    ds = ds.map(data_augmentation(training, image_size), num_parallel_calls=autotune).batch(batch_size)
    if training:
        ds = ds.map(lambda _x, _y: (data_augmentation_seq(_x, training=True), _y),
                    num_parallel_calls=autotune)
    ds = ds.prefetch(autotune)
    return ds
