import tensorflow as tf
from detection.anchor_boxes import LabelEncoder
from utils.box_utils import corners_to_center, center_to_corners, calc_IoU
from detection.losses import create_focal_loss, create_loc_loss


def test_corner_to_center():
    corners = tf.convert_to_tensor([
        [0.3, 0.2, 0.8, 0.6],
        [0.1, 0.3, 0.8, 0.5],
    ], dtype=tf.float64)

    centers = corners_to_center(corners)
    centers_expected = tf.convert_to_tensor([
        [0.55, 0.4, 0.5, 0.4],
        [0.45, 0.4, 0.7, 0.2],
    ], dtype=tf.float64)
    print("=== centers:\n", centers)
    print("=== expected:\n", centers_expected)


def test_center_to_corner():
    centers = tf.convert_to_tensor([
        [0.55, 0.4, 0.5, 0.4],
        [0.45, 0.4, 0.7, 0.2],
    ], dtype=tf.float64)

    corners = center_to_corners(centers)
    corners_expected = tf.convert_to_tensor([
        [0.3, 0.2, 0.8, 0.6],
        [0.1, 0.3, 0.8, 0.5],
    ], dtype=tf.float64)
    print(tf.equal(corners, corners_expected))
    print("=== corners:\n", corners)
    print("=== expected:\n", corners_expected)


def test_calc_IoU():
    boxes1_corners = tf.convert_to_tensor([
        [0.3, 0.2, 0.8, 0.6],
        [0.1, 0.4, 0.8, 0.7]
    ])
    boxes2_corners = tf.convert_to_tensor([
        [0.1, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.5, 0.9]
    ])
    expected = tf.convert_to_tensor([0.02 / 0.24, 0.09 / 0.3])
    iou = calc_IoU(boxes1_corners, boxes2_corners, mode='corner')
    print("=== corners:\n", iou)
    print("=== expected:\n", expected)


class LossTest:
    def __init__(self):
        self.focal_loss = create_focal_loss(3)
        self.l1_smooth_loss = create_loc_loss(3)

    def test_focal_loss(self):
        pass

    def test_l1_smooth_loss(self):
        pass


class AnchorBoxTest:
    def __init__(self):
        pass

    def test_total_dims(self):
        pass

    def test_gen_anchor_box(self):
        pass


class LabelEncoderTest:
    def __init__(self):
        self.encoder = LabelEncoder(num_classes=3)

    def test_matching(self):
        pass

    def test_compute_offsets(self):
        pass
