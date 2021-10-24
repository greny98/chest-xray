import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import densenet
from tensorflow.keras.utils import Sequence
import albumentations as augment
import cv2

from detection.anchor_boxes import LabelEncoder
from static_values.values import IMAGE_SIZE, BATCH_SIZE, l_diseases, object_names

autotune = tf.data.AUTOTUNE


def data_augmentation(training=False):
    if training:
        transform = augment.Compose([
            augment.ImageCompression(quality_lower=80, quality_upper=100, p=0.25),
            augment.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            augment.Rotate(limit=75, p=0.4),
            augment.RandomScale(0.15),
            augment.RandomCrop(800, 800, p=0.25),
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
        aug_img = tf.numpy_function(func=data_augmentation(training), inp=[image_file], Tout=tf.float32)
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


class ClassificationGenerator(Sequence):
    def __init__(self, image_files, labels, image_dir, batch_size=BATCH_SIZE, training=True):
        """
        Khởi tạo data generator cho bài toán phân loại
        :param image_files: ["abc.png", "cde.png"]
        :param labels: [ [0], [1], [0] ]
        :param image_dir:
        :param batch_size:
        :param training:
        """
        self.image_files = [os.path.join(image_dir, filename) for filename in image_files]
        self.labels = labels
        self.num_samples = len(self.image_files)
        self.steps = len(image_files) // batch_size
        self.indices = np.arange(0, self.num_samples, dtype=np.int)
        if len(image_files) % batch_size > 0:
            self.steps += 1
        self.batch_size = batch_size
        self.training = training
        if training:
            self.transform = augment.Compose([
                augment.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                augment.RandomRotate90(p=0.3),
                augment.RandomScale(0.15),
                augment.RandomCrop(800, 800, p=0.25),
                augment.Resize(IMAGE_SIZE, IMAGE_SIZE),
            ])
            self.on_epoch_end()
        else:
            self.transform = augment.Compose([augment.Resize(IMAGE_SIZE, IMAGE_SIZE)])

    def __getitem__(self, item):
        start = item * self.batch_size
        end = (item + 1) * self.batch_size
        selected = self.indices[start:end]
        tensor_inp = []
        diseases = [[] for _ in range(len(l_diseases))]
        for i in selected:
            processed_image = self.process_image(self.image_files[i])
            tensor_inp.append(processed_image)
            for i_res, res in enumerate(self.labels[i]):
                diseases[i_res].append(res)
        diseases = tuple([tf.convert_to_tensor(disease, dtype=tf.float32) for disease in diseases])
        tensor_inp = tf.convert_to_tensor(tensor_inp, dtype=tf.float32)
        tensor_inp = densenet.preprocess_input(tensor_inp)
        return tensor_inp, diseases

    def process_image(self, image_name):
        image = cv2.imread(image_name)
        return self.transform(image=image)['image']

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.training:
            np.random.shuffle(self.indices)


# ======================================================================================================================
class DetectionGenerator(Sequence):
    def __init__(self, images_info: dict, image_dir, num_classes=len(object_names), batch_size=BATCH_SIZE,
                 training=True):
        """
        Khởi tạo data generator cho bài toán phân loại
        :param image_files: ["abc.png", "cde.png"]
        :param labels: [ [0], [1], [0] ]
        :param image_dir:
        :param batch_size:
        :param training:
        """
        self.image_files = [os.path.join(image_dir, filename) for filename in images_info.keys()]
        self.label_encoder = LabelEncoder(num_classes)
        self.bboxes = [image['bboxes'] for image in images_info.values()]
        self.labels = [image['labels'] for image in images_info.values()]
        self.num_samples = len(self.image_files)
        self.steps = self.num_samples // batch_size
        self.indices = np.arange(0, self.num_samples, dtype=np.int)
        if self.num_samples % batch_size > 0:
            self.steps += 1
        self.batch_size = batch_size
        self.training = training
        if training:
            self.transform = augment.Compose([
                augment.RandomBrightnessContrast(),
                augment.RandomRotate90(p=0.3),
                augment.RandomScale(0.15),
                augment.RandomCrop(800, 800, p=0.25),
                augment.Resize(IMAGE_SIZE, IMAGE_SIZE),
            ], bbox_params=augment.BboxParams(format='coco'))
        else:
            self.transform = augment.Compose([augment.Resize(IMAGE_SIZE, IMAGE_SIZE)])

    def __getitem__(self, item):
        start = item * self.batch_size
        end = (item + 1) * self.batch_size
        selected = self.indices[start:end]
        tensor_inp = []
        labels = []
        offsets = []
        for i in selected:
            processed_image, trans_bboxes = self.process_image(self.image_files[i], self.bboxes[i], self.labels[i])
            tensor_inp.append(processed_image)
            label = tf.convert_to_tensor(self.labels[i])
            trans_bboxes = tf.convert_to_tensor(trans_bboxes, dtype=tf.float32)
            offset, label_oh = self.label_encoder.matching(trans_bboxes, label)
            offsets.append(offset)
            labels.append(label_oh)
        tensor_inp = tf.convert_to_tensor(tensor_inp, dtype=tf.float32)
        offsets = tf.convert_to_tensor(offsets, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        print(offsets)
        return densenet.preprocess_input(tensor_inp), (offsets, labels)

    def process_image(self, image_name, bboxes, labels):
        trans_bboxes = []
        for i, bbox in enumerate(bboxes):
            trans_bbox = list(bbox)
            trans_bbox.append(object_names[labels[i] - 1])
            trans_bboxes.append(trans_bbox)
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        img_h, img_w, _ = image.shape
        transformed = self.transform(image=image, bboxes=trans_bboxes)
        bboxes_transformed = []
        for x, y, w, h, _ in transformed['bboxes']:
            cx = (x + w / 2) / IMAGE_SIZE
            cy = (y + h / 2) / IMAGE_SIZE
            w = w / IMAGE_SIZE
            h = h / IMAGE_SIZE
            bboxes_transformed.append([cx, cy, w, h])
        return transformed['image'], bboxes_transformed

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.training:
            np.random.shuffle(self.indices)
