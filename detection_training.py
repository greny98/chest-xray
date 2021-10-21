from detection.anchor_boxes import AnchorBoxes
from detection.feature_pyramid import get_backbone, FeaturePyramid

# backbone = get_backbone(weights='ckpt/checkpoint')
# pyramid = FeaturePyramid(backbone)
anchor_boxes = AnchorBoxes(steps=[56, 28, 14, 7, 3, 1])
print(anchor_boxes.get_total_boxes())
