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
        arg_max_iou = tf.argmax(iou_anchor2gt, axis=1)
        # Label for anchor boxes
        anchor_box_labels = []
        for label, iou in zip(arg_max_iou, iou_anchor2gt):
            best_iou = tf.gather(iou, label)
            if best_iou >= iou_threshold:
                anchor_box_labels.append(to_categorical(gt_classes[label], self.num_classes))
            else:
                anchor_box_labels.append(to_categorical(0, self.num_classes))
        # TODO: Tính offsets cho anchor box từ ground truth and return both classes and offsets
