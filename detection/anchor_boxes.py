import numpy as np


class AnchorBoxes:
    def __init__(self, steps):
        self.steps = steps
        self.feature_widths = [1. / step for step in self.steps]
        self.aspect_ratios = [0.5, 1., 2.]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

    def get_total_boxes(self):
        total_boxes = np.array(self.steps)
        total_boxes = np.sum(total_boxes ** 2)
        n_ar = len(self.aspect_ratios)
        n_scale = len(self.scales)
        return total_boxes * n_ar * n_scale * 4

    def get_anchor_boxes(self):
        boxes = []
        for n_step in self.steps:
            f_num = 0
            for i in range(n_step):
                cx = (i + 0.5) * self.feature_widths[f_num]
                cy = (i + 0.5) * self.feature_widths[f_num]
                for scale in self.scales:
                    for ratio in self.aspect_ratios:
                        sqrt_ratio = np.sqrt(ratio)
                        width = scale * sqrt_ratio * self.feature_widths[f_num]
                        height = scale * (1 / sqrt_ratio) * self.feature_widths[f_num]
                        boxes.append([cx, cy, width, height])
        return boxes


class LabelEncoder:
    def __init__(self):
        self._anchor_boxes = AnchorBoxes(steps=[56, 28, 14, 7, 3, 1])
