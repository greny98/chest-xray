from detection.anchor_boxes import LabelEncoder, PredictionDecoder
from static_values.values import IMAGE_SIZE
from utils.dataframe import read_csv
from utils.data_generator import DetectionGenerator
from cv2 import cv2
import tensorflow as tf

image_infos = read_csv('data/train_bbox.csv', mode='detect')
image_file = list(image_infos.keys())[0]

img = cv2.imread(f'data/images/{image_file}')
img_h, img_w, _ = img.shape
gt_boxes = image_infos[image_file]['bboxes']
for x, y, w, h in gt_boxes:
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

image_infos_test = {}
image_infos_test[image_file] = image_infos[image_file]

label_encoder = LabelEncoder()

ds = DetectionGenerator(image_infos_test, 'data/images', label_encoder, training=True, batch_size=1)
img_test = None
bboxes_test = None
for b_img_test, bboxes in ds:
    img_test = b_img_test
    bboxes_test = bboxes
    break

x, y, w, h = tf.reshape(bboxes_test, shape=(-1,)).numpy()
x = int(x * IMAGE_SIZE)
y = int(y * IMAGE_SIZE)
w = int(w * IMAGE_SIZE)
h = int(h * IMAGE_SIZE)

img_test = tf.cast(img_test, tf.uint8)
img_test = img_test.numpy()[0, :, :, :]
cv2.rectangle(img_test, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.imshow("img_test", img_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
