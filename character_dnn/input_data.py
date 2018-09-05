# coding=utf-8
"""Functions for reading character_dnn data."""
import os

import numpy as np


def load_data_label(base_model_dir):
    train_vec_filename = os.path.join(base_model_dir, "../data/vec/emotion_train_vec.npy")
    train_label_filename = os.path.join(base_model_dir, '../data/label/train_label.npy')
    test_vec_filename = os.path.join(base_model_dir, '../data/vec/emotion_test_vec.npy')
    test_label_filename = os.path.join(base_model_dir, '../data/label/test_label.npy')

    X_train = np.load(train_vec_filename)
    print('X_train', X_train.shape)
    Y_train = np.load(train_label_filename)
    print('Y_train', Y_train.shape)
    X_test = np.load(test_vec_filename)
    print('X_test', X_test.shape)
    Y_test = np.load(test_label_filename)
    print('Y_test', Y_test.shape)
    return X_train, Y_train, X_test, Y_test

def load_data_label_combine(X_train, X_test, X1_train, X1_test):
    """
    列向合并矩阵
    combine two arr into one
    :return:
    """
    X_train_all = np.hstack((X_train, X1_train))
    X_test_all = np.hstack((X_test, X1_test))
    return X_train_all, X_test_all


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data_label('')
    print(X_test)
    print(Y_test)
