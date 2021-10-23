import numpy as np

from static_values.values import l_diseases
from utils.data_generator import ClassificationGenerator
from utils.dataframe import train_val_split, read_csv

X_train_val_df, y_train_val_df = read_csv('data/train_val.csv')
# Split train, val
(X_train, y_train), (X_val, y_val) = train_val_split(X_train_val_df.values, y_train_val_df.values, log=False)
# Flatten X
X_train = X_train.reshape(-1)
X_val = X_val.reshape(-1)

ds = ClassificationGenerator(X_train, y_train, image_dir='data/images')

for batch in ds:
    print(batch)
    break
