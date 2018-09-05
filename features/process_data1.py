import os
from collections import defaultdict
import re
import csv

import numpy as np
import pandas as pd


def build_data_cv(datafile, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)

    with open(datafile, "rb") as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        first_line = True
        for line in csvreader:
            if first_line:
                first_line = False
                continue
            status = []
            sentences = re.split(r'[.?]', line[1].strip())
            try:
                sentences.remove('')
            except ValueError:
                None

            for sent in sentences:
                if clean_string:
                    orig_rev = clean_str(sent.strip())
                    if orig_rev == '':
                        continue
                    words = set(orig_rev.split())
                    splitted = orig_rev.split()
                    if len(splitted) > 150:
                        orig_rev = []
                        splits = int(np.floor(len(splitted) / 20))
                        for index in range(splits):
                            orig_rev.append(' '.join(splitted[index * 20:(index + 1) * 20]))
                        if len(splitted) > splits * 20:
                            orig_rev.append(' '.join(splitted[splits * 20:]))
                        status.extend(orig_rev)
                    else:
                        status.append(orig_rev)
                else:
                    orig_rev = sent.strip().lower()
                    words = set(orig_rev.split())
                    status.append(orig_rev)

                for word in words:
                    vocab[word] += 1

            datum = {"y0": 1 if line[2].lower() == 'y' else 0,
                     "y1": 1 if line[3].lower() == 'y' else 0,
                     "y2": 1 if line[4].lower() == 'y' else 0,
                     "y3": 1 if line[5].lower() == 'y' else 0,
                     "y4": 1 if line[6].lower() == 'y' else 0,
                     "text": status,
                     "user": line[0],
                     "num_words": np.max([len(sent.split()) for sent in status]),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)

    return revs, vocab


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    #    string = re.sub(r"[a-zA-Z]{4,}", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == "__main__":
    base_dir = '/data/essays_data'

    data_folder = os.path.join(base_dir, 'essays.csv')
    print("loading data...")
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    num_words = pd.DataFrame(revs)["num_words"]
    max_l = np.max(num_words)
    print("data loaded!")
    print("number of status: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
