from siim.data_generator import create_images_info
from utils.data_generator import DetectionGenerator
from detection.anchor_boxes import LabelEncoder

image_info = create_images_info('data/siim/train_val.csv')

# label_encoder = LabelEncoder()
# train_ds = DetectionGenerator(info, '../data/siim/images', label_encoder=label_encoder, training=True,
#                               batch_size=8)
# for img, labels in train_ds:
#     print(labels)
#     break

for image, info in image_info.items():
    bboxes = info['bboxes']
    for x, y, w, h in bboxes:
        if x < 0 or y < 0 or w < 0 or h < 0:
            print(x, y, w, h)
            print(image)
