import tensorflow as tf


def swap_xy(boxes):
    return tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)


def corners_to_center(corners):
    """
    Convert boxes from (xmin,ymin,xmax,ymax) to (x,y,w,h)
    :param corners:
    :return:
    """
    center = (corners[..., :2] + corners[..., 2:]) / 2.0
    size = corners[..., 2:] - corners[..., :2]
    return tf.concat([center, size], axis=-1)


def center_to_corners(center):
    """
    Convert boxes from (x,y,w,h) to (xmin,ymin,xmax,ymax)
    :param center:
    :return:
    """
    top_left = center[..., :2] - center[..., 2:] / 2.0
    bottom_right = center[..., :2] + center[..., 2:] / 2.0
    return tf.concat([top_left, bottom_right], axis=-1)


def calc_IoU(anchors, gt_boxes, mode='corner', reduce_mean=False):
    """
    Compute IoU of predictions and ground_truth
        (Use for corners)
    :param anchors:
    :param gt_boxes:
    :param mode:
    :param reduce_mean:
    :return:
    """
    if mode == 'center':
        anchors = center_to_corners(anchors)
        gt_boxes = center_to_corners(gt_boxes)

    # Calculate Intersection
    inter_coor_min = tf.maximum(anchors[:, None, :2], gt_boxes[:, :2])
    inter_coor_max = tf.minimum(anchors[:, None, 2:], gt_boxes[:, 2:])
    inter = tf.maximum(0., inter_coor_max - inter_coor_min)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    # Calculate Union
    anchors_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    gt_boxes_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union_area = tf.maximum(anchors_area[:, None] + gt_boxes_area - inter_area, 1e-8)

    # IoU
    IoU = tf.clip_by_value(inter_area / union_area, 0., 1.)
    if reduce_mean:
        return tf.reduce_mean(IoU)
    return IoU
