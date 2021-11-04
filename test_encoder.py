import tensorflow as tf
from detection.anchor_boxes import LabelEncoder
from utils.data_generator import DetectionGenerator
from utils.dataframe import read_csv

image_infos = read_csv('data/train_bbox.csv', mode='detect')
label_encoder = LabelEncoder()
ds = DetectionGenerator(image_infos, 'data/images', label_encoder, training=True, batch_size=4)

for batch in ds:
    print(batch[0].shape)
    break
