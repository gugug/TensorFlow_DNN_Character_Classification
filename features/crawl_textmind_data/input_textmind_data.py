# coding=utf-8
"""读取文件"""
import os

from tensorflow.python.platform import gfile

__author__ = 'gu'

import numpy as np


def load_corpus():
    """
    加载训练数据
    :return:
    """
    base_dir = 'E:\Koo\Projects\PycharmProjects\TensorFlow_DNN_Character_Classification\data\essay_data'

    train_txt_path = os.path.join(base_dir, "vocab1_train.txt")
    test_txt_path = os.path.join(base_dir, "vocab1_test.txt")

    return read_lines(train_txt_path), read_lines(test_txt_path)


def read_lines(train_txt_path):
    with gfile.Open(train_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line)
        print('txt lines length', len(lines))
        return lines


def load_textmind_data_label(base_model_dir):
    """
    加载textmind矩阵
    :param base_model_dir:
    :return:
    """
    textmind_train_vec = "textmind_train_vec.npy"
    textmind_train_label = "train_label.npy"
    textmind_test_vec = "textmind_test_vec.npy"
    textmind_test_label = "test_label.npy"

    train_vec_filename = os.path.join(base_model_dir, textmind_train_vec)
    train_label_filename = os.path.join(base_model_dir, textmind_train_label)
    test_vec_filename = os.path.join(base_model_dir, textmind_test_vec)
    test_label_filename = os.path.join(base_model_dir, textmind_test_label)

    X_train = np.load(train_vec_filename)
    print('X_train', X_train.shape)
    Y_train = np.load(train_label_filename)
    print('Y_train', Y_train.shape)
    X_test = np.load(test_vec_filename)
    print('X_test', X_test.shape)
    Y_test = np.load(test_label_filename)
    print('Y_test', Y_test.shape)
    return X_train, Y_train, X_test, Y_test


def load_textmind_data_label_with_normalization(base_model_dir):
    """
    加载textmind矩阵 并进行数据平滑
    :param base_model_dir:
    :return:
    """
    textmind_train_vec = "textmind_train_vec.npy"
    textmind_train_label = "train_label.npy"
    textmind_test_vec = "textmind_test_vec.npy"
    textmind_test_label = "test_label.npy"

    train_vec_filename = os.path.join(base_model_dir, textmind_train_vec)
    train_label_filename = os.path.join(base_model_dir, textmind_train_label)
    test_vec_filename = os.path.join(base_model_dir, textmind_test_vec)
    test_label_filename = os.path.join(base_model_dir, textmind_test_label)

    X_train = np.load(train_vec_filename)
    print('X_train', X_train.shape)
    Y_train = np.load(train_label_filename)
    print('Y_train', Y_train.shape)
    X_test = np.load(test_vec_filename)
    print('X_test', X_test.shape)
    Y_test = np.load(test_label_filename)
    print('Y_test', Y_test.shape)
    X_train_1 = np.where(X_train >= 0, np.log(X_train + 1), 0)
    X_test_1 = np.where(X_test >= 0, np.log(X_test + 1), 0)
    return X_train_1, Y_train, X_test_1, Y_test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_textmind_data_label_with_normalization('')
    print(X_test)
    print(Y_test)
