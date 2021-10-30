import numpy as np

from detection.anchor_boxes import LabelEncoder, PredictionDecoder
import tensorflow as tf

from detection.ssd import create_ssd_model
from utils.box_utils import calc_IoU
from utils.dataframe import read_csv
from cv2 import cv2
from tensorflow.keras.applications import densenet

image_infos = read_csv('data/train_bbox.csv', mode='detect')
image_file = list(image_infos.keys())[1]

image_infos_test = {}
image_infos_test[image_file] = image_infos[image_file]

img = cv2.imread(f'data/images/{image_file}')
img_h, img_w, _ = img.shape
gt_boxes = image_infos[image_file]['bboxes']
print(image_infos[image_file])

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

ssd_model = create_ssd_model()
ssd_model.load_weights('ckpt/detect_v5/checkpoint')
pred_labels, pred_offsets = ssd_model(img_tensor, training=False)

results = decoder.compute_bboxes(pred_offsets, pred_labels)
bboxes = results[0]["bboxes"].numpy()
labels = results[0]["labels"].numpy()
print(results)
print(np.array(gt_boxes) / img_w)

gt = tf.convert_to_tensor(gt_boxes) / img_w

iou = calc_IoU(gt, bboxes, mode='center')
argmax = tf.argmax(iou).numpy()
print("iou: ", tf.reduce_max(iou))

print(labels[argmax])
print(bboxes[argmax, :])

# cp_img = img[:, :, :]
# cp_img = cv2.resize(cp_img, (320, 320))
# for cx, cy, w, h in bboxes:
#     x = cx - 0.5 * w
#     y = cy - 0.5 * h
#     x = int(x * 320)
#     y = int(y * 320)
#     w = int(w * 320)
#     h = int(h * 320)
#     cv2.rectangle(cp_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("img", img)
# cv2.imshow("cp_img", cp_img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
