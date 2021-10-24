import pandas as pd
import albumentations as A
import cv2

# train_df = pd.read_csv('data/train_bbox.csv')
# row = train_df.loc[0, :]
# image = cv2.imread('data/images/' + row['image_idx'])
# bboxes = row[['left_box', 'top_']]
#
#
# transform = A.Compose([
#     A.RandomRotate90(),
#     A.RandomBrightnessContrast(p=0.2),
# ], bbox_params=A.BboxParams(format='coco'))
#
# transformed = transform(image=image, bboxes=bboxes)
# transformed_image = transformed['image']
# transformed_bboxes = transformed['bboxes']
from utils.data_generator import DetectionGenerator
from utils.dataframe import read_csv

images_info = read_csv('data/train_bbox.csv', mode='detect')
train_ds = DetectionGenerator(images_info, 'data/images')

for batch in train_ds:
    print(batch)
    break
