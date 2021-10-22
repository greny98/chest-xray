from tensorflow.python.keras.applications import densenet

from utils import box_utils
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class AnchorBoxes:
    def __init__(self, steps):
        self.steps = steps
        self.feature_widths = [1. / step for step in self.steps]
        self.aspect_ratios = [0.5, 1., 2.]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
        self.boxes = self.gen_anchor_boxes()
        self.total_dims = self.get_total_dims()

    def get_total_dims(self):
        total_boxes = tf.convert_to_tensor(self.steps, dtype=tf.int32)
        total_boxes = tf.reduce_sum(tf.square(total_boxes))
        n_ar = len(self.aspect_ratios)
        n_scale = len(self.scales)
        return total_boxes * n_ar * n_scale * 4

    def gen_anchor_boxes(self):
        boxes = []
        total = 0
        for f_num, n_step in enumerate(self.steps):
            for row in range(n_step):
                for col in range(n_step):
                    cx = (col + 0.5) * self.feature_widths[f_num]
                    cy = (row + 0.5) * self.feature_widths[f_num]
                    for scale in self.scales:
                        for ratio in self.aspect_ratios:
                            sqrt_ratio = tf.sqrt(ratio)
                            width = scale * sqrt_ratio * self.feature_widths[f_num]
                            height = scale * (1 / sqrt_ratio) * self.feature_widths[f_num]
                            boxes.append(tf.convert_to_tensor([cx, cy, width, height]))
                            total += 1
        return tf.stack(boxes, axis=0)


class LabelEncoder:
    def __init__(self, num_classes):
        self.anchor_boxes = AnchorBoxes(steps=[56, 28, 14, 7, 3, 1])
        self.num_classes = num_classes + 1  # 0 for background

    def compute_offsets(self, matched_gt_boxes):
        # sigma_center = 0.1
        # sigma_size = 0.2
        off_cx = (matched_gt_boxes[:, 0] - self.anchor_boxes.boxes[:, 0]) / self.anchor_boxes.boxes[:, 2]
        off_cy = (matched_gt_boxes[:, 1] - self.anchor_boxes.boxes[:, 1]) / self.anchor_boxes.boxes[:, 3]
        off_w = tf.math.log(matched_gt_boxes[:, 2] / self.anchor_boxes.boxes[:, 2])
        off_h = tf.math.log(matched_gt_boxes[:, 3] / self.anchor_boxes.boxes[:, 3])
        return tf.stack([off_cx, off_cy, off_w, off_h], axis=1)

    def matching(self, gt_boxes, gt_classes, iou_threshold=0.5):
        """
        Matching ground truth boxes and anchor boxes
        :param gt_boxes:
        :param gt_classes:
        :param iou_threshold:
        :return:
        """
        iou_anchor2gt = []
        for idx, gt_box in enumerate(gt_boxes):
            gt_box_cen = box_utils.center_to_corners(tf.expand_dims(gt_box, axis=0))
            anchor_cen = box_utils.center_to_corners(self.anchor_boxes.boxes)
            iou = box_utils.calc_IoU(gt_box_cen, anchor_cen)
            iou_anchor2gt.append(iou)
        iou_anchor2gt = tf.stack(iou_anchor2gt, axis=1)
        # Get best IoU
        rows = tf.range(0, self.anchor_boxes.total_dims / 4, dtype=tf.int32)
        arg_max_iou = tf.argmax(iou_anchor2gt, axis=1)
        arg_max_iou = tf.cast(arg_max_iou, tf.int32)
        best_iou = tf.gather_nd(
            iou_anchor2gt,
            tf.stack([rows, arg_max_iou], axis=1))
        # TODO: Labeled for anchor boxes
        # Find anchor boxes that has iou >= iou_threshold
        negative_indices = tf.cast(tf.where(best_iou < iou_threshold), tf.int32)
        anchor_boxes_classes = tf.tensor_scatter_nd_update(
            tf.expand_dims(tf.gather(gt_classes, arg_max_iou), axis=1),
            negative_indices,
            tf.zeros_like(negative_indices))
        anchor_boxes_classes = tf.reshape(anchor_boxes_classes, shape=(-1,))
        matched_gt_boxes = tf.gather_nd(gt_boxes, tf.expand_dims(arg_max_iou, axis=-1))
        # TODO: Compute offsets for anchor box from ground truth and return both classes and offsets
        offsets = self.compute_offsets(matched_gt_boxes)
        return offsets, to_categorical(anchor_boxes_classes, self.num_classes)
