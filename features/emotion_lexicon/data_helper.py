import os
from collections import defaultdict
import re
import csv

import numpy as np
import pandas as pd


def build_emotion_lexicon_dict(datafile):
    """
    load emotion csv
    """
    print('loading emotion dict...')
    vocab_emotion_dict = defaultdict(float)
    with open(datafile, "rb") as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        first_line = True
        for line in csvreader:
            if first_line:
                first_line = False
                continue
            status = []
            # print(line)
            try:
                line.remove('')
            except ValueError:
                None
            word = line[0]
            orig_rev = word.strip().lower()
            status.append(orig_rev)
            # print(orig_rev)
            word_emotion_value = []
            for value in line[1:]:
                word_emotion_value.append(1 if value == '1' else 0)
            # print(word_emotion_value)
            vocab_emotion_dict[orig_rev] = word_emotion_value
    print('emotion word size are %s ' % len(vocab_emotion_dict))
    return vocab_emotion_dict


def build_emotion_feature(filename, vocab_emotion_dict):
    """
    build emotion feature
    """
    X_input = []
    y_output = []
    with open(filename, "rb") as f:
        lines = f.readlines()
        for line in lines:
            text = line.strip().split()
            y = [1 if s == '1' else 0 for s in text[1:6]]
            emotion_value = []
            for word in text[6:]:
                if vocab_emotion_dict.__contains__(word):
                    word_emotion_values = vocab_emotion_dict[word]
                    emotion_value.append(word_emotion_values)
            value = map(sum, zip(*emotion_value))
            X_input.append(value)
            y_output.append(y)
    X_input = np.mat(X_input)
    y_output = np.mat(y_output)
    print('X_input.shape', X_input.shape)
    print('y_output.shape', y_output.shape)
    return X_input, y_output


def save_arr(filename, X_sp):
    np.save(filename, X_sp)
    print('write done', filename)


def load_data_label():
    """
    load and save arr
    :return:
    """
    base_dir = ''
    data_folder = os.path.join(base_dir, 'Emotion_Lexicon.csv')
    print "loading data...",
    emotion_dict = build_emotion_lexicon_dict(data_folder)
    X_train, y_train = build_emotion_feature('../data/essay_data/vocab1_train.txt', emotion_dict)
    X_test, y_test = build_emotion_feature('../data/essay_data/vocab1_test.txt', emotion_dict)
    save_arr('emotion_train_vec.npy', X_train)
    save_arr('emotion_train_label.npy', y_train)
    save_arr('emotion_test_vec.npy', X_test)
    save_arr('emotion_test_label.npy', y_test)
    return X_train, y_train, X_test, y_test


def load_emotion_data_label(base_model_dir):
    """
    load .npy file for arr
    :param base_model_dir:
    :return:
    """
    train_vec_filename = os.path.join(base_model_dir, "emotion_train_vec.npy")
    train_label_filename = os.path.join(base_model_dir, 'emotion_train_label.npy')
    test_vec_filename = os.path.join(base_model_dir, 'emotion_test_vec.npy')
    test_label_filename = os.path.join(base_model_dir, 'emotion_test_label.npy')
    X_train = np.load(train_vec_filename)
    print('X_train', X_train.shape)
    Y_train = np.load(train_label_filename)
    print('Y_train', Y_train.shape)
    X_test = np.load(test_vec_filename)
    print('X_test', X_test.shape)
    Y_test = np.load(test_label_filename)
    print('Y_test', Y_test.shape)
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    # load_data_label()
    load_emotion_data_label('')
