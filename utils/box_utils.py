import tensorflow as tf


def swap_xy(boxes):
    return tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]], axis=-1)


def corners_to_center(corners):
    """
    Convert boxes from (xmin,ymin,xmax,ymax) to (x,y,w,h)
    :param corners:
    :return:
    """
    x = (corners[..., 0] + corners[..., 2]) * 0.5
    y = (corners[..., 1] + corners[..., 3]) * 0.5
    w = corners[..., 2] - corners[..., 0]
    h = corners[..., 3] - corners[..., 1]
    return tf.stack([x, y, w, h], axis=-1)


def center_to_corners(center):
    """
    Convert boxes from (x,y,w,h) to (xmin,ymin,xmax,ymax)
    :param center:
    :return:
    """
    xmin = center[..., 0] - center[..., 2] / 2
    ymin = center[..., 1] - center[..., 3] / 2
    xmax = center[..., 0] + center[..., 2] / 2
    ymax = center[..., 1] + center[..., 3] / 2
    return tf.stack([xmin, ymin, xmax, ymax], axis=-1)


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
    union_area = tf.maximum(anchors_area[:, None] + gt_boxes_area - inter_area, 1e-7)

    # IoU
    IoU = tf.clip_by_value(inter_area / union_area, 0., 1.)
    if reduce_mean:
        return tf.reduce_mean(IoU)
    return IoU
