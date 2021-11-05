# List of diseases
l_diseases = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
              'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',
              'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',
              'Consolidation']

# Image size default
# IMAGE_SIZE = 512
IMAGE_SIZE = 1024

# Batch size
BATCH_SIZE = 9

object_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass',
                'Nodule', 'Pneumonia', 'Pneumothorax']
object_names2idx = {key: idx for idx, key in enumerate(object_names)}
