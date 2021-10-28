# List of diseases
l_diseases = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration',
              'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',
              'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',
              'Consolidation']

# Image size default
IMAGE_SIZE = 320

# Batch size
BATCH_SIZE = 3

object_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass',
                'Nodule', 'Pneumonia', 'Pneumothorax']
object_names2idx = {key: idx + 1 for idx, key in enumerate(object_names)}

STEPS = [40, 20, 10, 5, 2, 1]