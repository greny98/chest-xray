import numpy as np

from detection.anchor_boxes import PredictionDecoder
import tensorflow as tf
from tensorflow.keras import Model
from detection.ssd import create_ssd_model
from utils import box_utils
from utils.dataframe import read_csv
from cv2 import cv2
from tensorflow.keras.applications import densenet

image_infos = read_csv('data/train_bbox.csv', mode='detect')
image_file = list(image_infos.keys())[100]

image_infos_test = {}
image_infos_test[image_file] = image_infos[image_file]

img = cv2.imread(f'data/images/{image_file}')
img_h, img_w, _ = img.shape
gt_boxes = image_infos[image_file]['bboxes']
print("=== image_infos[image_file]:", image_infos[image_file])
gt_boxes = box_utils.center_to_corners(tf.convert_to_tensor(gt_boxes, tf.float32))

# img_tensor = cv2.resize(img, (640, 640))
img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
img_tensor = tf.expand_dims(img_tensor, axis=0)
img_tensor = densenet.preprocess_input(img_tensor)

model = create_ssd_model()
model.load_weights('ckpt/detect_densenet_v1/checkpoint')
pred_labels = model(img_tensor, training=False)

image = tf.keras.Input(shape=(1024, 1024, 3,), name="image")
predictions = model(image, training=False)
detections = PredictionDecoder()(image, predictions)

inference_model = Model(inputs=image, outputs=detections)
bboxes, scores, labels, _ = inference_model.predict(img_tensor)

print("bboxes", tf.convert_to_tensor(bboxes), gt_boxes)
print("scores, labels", scores, labels)
iou = box_utils.calc_IoU(gt_boxes, tf.convert_to_tensor(bboxes[0]))
print(tf.reduce_max(iou))
