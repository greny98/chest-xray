from detection.feature_pyramid import get_backbone, FeaturePyramid
from detection.ssd import create_ssd_model

backbone = get_backbone()
feature_pyramid = FeaturePyramid(backbone)
ssd = create_ssd_model()
ssd.summary()
