import numpy as np

from detection.anchor_boxes import LabelEncoder, PredictionDecoder
import tensorflow as tf
from tensorflow.keras import Model
from detection.ssd import create_ssd_model
from utils import box_utils
from utils.dataframe import read_csv
from cv2 import cv2
from tensorflow.keras.applications import densenet

image_infos = read_csv('data/train_bbox.csv', mode='detect')
image_file = list(image_infos.keys())[3]

image_infos_test = {}
image_infos_test[image_file] = image_infos[image_file]

img = cv2.imread(f'data/images/{image_file}')
img_h, img_w, _ = img.shape
gt_boxes = image_infos[image_file]['bboxes']
gt_boxes = np.array(gt_boxes) / img_h

for x, y, w, h in gt_boxes:
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

label_encoder = LabelEncoder()
decoder = PredictionDecoder(label_encoder.anchor_boxes)

img_tensor = cv2.resize(img, (512, 512))
img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
img_tensor = tf.expand_dims(img_tensor, axis=0)
img_tensor = densenet.preprocess_input(img_tensor)

model = create_ssd_model()
model.load_weights('ckpt/detect_densenet/checkpoint')
pred_labels = model(img_tensor, training=False)

image = tf.keras.Input(shape=(512, 512, 3,), name="image")
predictions = model(image, training=False)
detections = PredictionDecoder(label_encoder.anchor_boxes)(predictions)

inference_model = Model(inputs=image, outputs=detections)
results = inference_model.predict(img_tensor)
bboxes, scores, labels, n_valid = results
gt_boxes = box_utils.center_to_corners(tf.convert_to_tensor(gt_boxes, tf.float32))
print(bboxes, gt_boxes, scores, labels)
iou = box_utils.calc_IoU(bboxes[0, :, :], gt_boxes)
print(iou)
