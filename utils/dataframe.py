from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
import pandas as pd

from static_values.values import l_diseases, object_names2idx


def split_labels(df: pd.DataFrame):
    """
    Split to Image column and Diseases columns
    :param df:
    :return:
    """
    X_col = df[['Image Index']]
    y_cols = df[l_diseases]
    return X_col, y_cols


def create_bboxes(df):
    images = df['image_idx'].unique()
    images_dict = {key: {} for key in images}
    for image in images_dict.keys():
        image_box_df = df[df['image_idx'] == image]
        images_dict[image] = {
            "bboxes": image_box_df[['box_left', 'box_top',
                                    'box_width', 'box_height']].values,
            "labels": [object_names2idx[label]
                       for label in image_box_df['finding_label'].values]
        }
    return images_dict


def read_csv(filename: str, mode='classify'):
    df = pd.read_csv(filename)
    if mode == 'classify':
        return split_labels(df)
    elif mode == 'detect':
        return create_bboxes(df)


def train_val_split(X_train_val, y_train_val, test_size=0.1, log=False):
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_train_val, y_train_val, test_size=test_size)
    if log:
        print("Split information:")
        pos_ratio = y_train.sum(axis=0) / y_val.sum(axis=0)
        pos_train_ratio = np.mean(y_train, axis=0)
        pos_val_ratio = np.mean(y_val, axis=0)
        print("- Ratio:")
        for i, disease in enumerate(l_diseases):
            print(
                f"\t+ {disease}: train/val={pos_ratio[i]} - pos_train: {pos_train_ratio[i]} - pos_val: {pos_val_ratio[i]}")
    return (X_train, y_train), (X_val, y_val)
