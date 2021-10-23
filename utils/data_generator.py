import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import densenet
from tensorflow.keras.utils import Sequence
import albumentations as augment
import cv2
from static_values.values import IMAGE_SIZE, BATCH_SIZE, l_diseases

autotune = tf.data.AUTOTUNE


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
                # augment.RandomSnow(p=0.25),
                # augment.RandomFog(p=0.25),
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
                augment.RandomBrightnessContrast(),
                augment.RandomRotate90(p=0.3),
                augment.RandomScale(0.15),
                augment.RandomSnow(p=0.25),
                augment.RandomFog(p=0.25),
                augment.RandomCrop(800, 800, p=0.25),
                augment.Resize(IMAGE_SIZE, IMAGE_SIZE),
            ])
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
        return densenet.preprocess_input(tensor_inp), diseases

    def process_image(self, image_name):
        image = cv2.imread(image_name)
        return self.transform(image=image)['image']

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.training:
            np.random.shuffle(self.indices)
