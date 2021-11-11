import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import densenet
import albumentations as augment

from detection.anchor_boxes import LabelEncoder
from static_values.values import IMAGE_SIZE, BATCH_SIZE, object_names

autotune = tf.data.AUTOTUNE


def classify_augmentation(training=False):
    if training:
        transform = augment.Compose([
            augment.ImageCompression(quality_lower=90, quality_upper=100, p=0.4),
            augment.RandomCrop(1000, 1000),
            augment.HorizontalFlip(p=0.3),
            augment.VerticalFlip(p=0.3),
            augment.RandomRotate90(p=0.3),
            augment.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            augment.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
            augment.GaussNoise(p=0.4),
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


def ClassifyGenerator(images, y, image_dir, training=False, batch_size=BATCH_SIZE):
    def process_data(image_file, label):
        aug_img = tf.numpy_function(func=classify_augmentation(training), inp=[image_file], Tout=tf.float32)
        return aug_img, label

    images_ts = tf.data.Dataset.from_tensor_slices(image_dir + images)
    labels_ts = []
    for col in range(y.shape[1]):
        labels_ts.append(tf.data.Dataset.from_tensor_slices(y[:, col].astype(float)))
    labels_ts = tf.data.Dataset.zip(tuple(labels_ts))
    ds = tf.data.Dataset.zip((images_ts, labels_ts))
    ds = ds.shuffle(24 * batch_size, reshuffle_each_iteration=training)
    ds = ds.map(lambda x, y: process_data(x, y),
                num_parallel_calls=autotune).batch(batch_size)
    ds = ds.prefetch(autotune)
    return ds


# ======================================================================================================================

def detect_augmentation(label_encoder: LabelEncoder, training: bool):
    if training:
        transform = augment.Compose([
            augment.ImageCompression(quality_lower=80, quality_upper=100),
            augment.HorizontalFlip(),
            augment.VerticalFlip(),
            augment.RandomRotate90(),
            augment.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            augment.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
            augment.GaussNoise(),
            augment.RandomSizedBBoxSafeCrop(640, 640),
        ], bbox_params=augment.BboxParams(format='coco'))
    else:
        transform = augment.Compose([augment.Resize(IMAGE_SIZE, IMAGE_SIZE)],
                                    bbox_params=augment.BboxParams(format='coco'))

    def preprocess_image(image_file, bboxes, labels, n_bbox):
        image_raw = tf.io.read_file(image_file)
        trans_bboxes = []
        bboxes = bboxes[:n_bbox]
        labels = labels[:n_bbox]
        for i, bbox in enumerate(bboxes[:n_bbox]):
            trans_bbox = list(bbox)
            trans_bbox.append(object_names[labels[i] - 1])
            trans_bboxes.append(trans_bbox)
        decoded = tf.image.decode_jpeg(image_raw, channels=3)
        data = {'image': decoded.numpy(), 'bboxes': trans_bboxes}
        transformed = transform(**data)
        # extract transformed image
        aug_img = transformed['image']
        aug_img = tf.cast(aug_img, tf.float32)
        aug_img = densenet.preprocess_input(aug_img)
        aug_img = tf.cast(aug_img, tf.float32)

        # extract transformed bboxes
        bboxes_transformed = []
        for x, y, w, h, _ in transformed['bboxes']:
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            bboxes_transformed.append(tf.convert_to_tensor([cx, cy, w, h], tf.float32))
        print(len(bboxes_transformed))
        bboxes_transformed = tf.convert_to_tensor(bboxes_transformed, tf.float32)
        labels = tf.convert_to_tensor(labels, tf.float32)
        labels = label_encoder.encode_sample(aug_img.shape, bboxes_transformed, labels)
        return [aug_img, labels]

    return preprocess_image


def DetectionGenerator(images_info: dict, image_dir, label_encoder, training=False,
                       batch_size=10):
    # Extract infomation from images_info
    image_files = [os.path.join(image_dir, filename) for filename in images_info.keys()]
    bboxes = [list(image['bboxes']) for image in images_info.values()]
    labels = [list(image['labels']) for image in images_info.values()]
    # padding boxes
    pad_bbox = np.zeros(4, dtype=np.float32)
    pad_label = -1
    num_bboxes = [len(label) for label in labels]
    max_padding = max(num_bboxes)
    for i in range(len(bboxes)):
        for _ in range(num_bboxes[i], max_padding):
            bboxes[i].append(pad_bbox)
            labels[i].append(pad_label)
    # Create tensor slices
    image_files_slices = tf.data.Dataset.from_tensor_slices(image_files)
    bboxes_slices = tf.data.Dataset.from_tensor_slices(bboxes)
    labels_slices = tf.data.Dataset.from_tensor_slices(labels)
    num_bboxes_slices = tf.data.Dataset.from_tensor_slices(num_bboxes)
    y_slices = tf.data.Dataset.zip((bboxes_slices, labels_slices, num_bboxes_slices))

    # Create dataset with process
    def process_data(image_file, y):
        aug_image, labels = tf.numpy_function(func=detect_augmentation(label_encoder, training),
                                              inp=[image_file, y[0], y[1], y[2]],
                                              Tout=[tf.float32, tf.float32])
        return aug_image, labels

    ds = tf.data.Dataset.zip((image_files_slices, y_slices))
    ds = ds.shuffle(256, reshuffle_each_iteration=training)
    ds = ds.map(lambda x, y: process_data(x, y), num_parallel_calls=autotune).batch(batch_size).prefetch(-1)
    return ds
