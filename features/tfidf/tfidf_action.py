# -*- coding: UTF-8 -*-
'''tfidf-lsi-svm stack for gender'''

from __future__ import division

import codecs
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from numpy import *
from gensim import models, corpora
import numpy as np

INPUT_SIZE = 300  # 训练特征维度参数


class user_predict:
    def __init__(self, train_document, text_document):
        self.train_document = train_document
        self.text_document = text_document

    # -----------------------准确值计算-----------------------
    def myAcc(self, y_true, y_pred):
        true_num = 0
        # 最大数的索引
        y_pred = np.argmax(y_pred, axis=1)

        # for i in range(y_true.__len__()):
        #     print y_true[i]
        for i in range(y_pred.__len__()):
            if y_true[i] == y_pred[i]:
                true_num += 1
        return true_num

    # -----------------------load data-----------------------
    def load_data(self, doc):
        list_name = []
        list_total = []
        list_gender = []
        # 对应标签导入词典
        f = codecs.open(doc)
        temp = f.readlines()
        print len(temp)

        for i in range(len(temp)):
            temp[i] = temp[i].split(" ")
            user_name = temp[i][0]
            tags = temp[i][1:6]
            query = temp[i][6:]
            query = " ".join(query).strip().replace("\n", "")
            list_total.append(query)
            list_gender.append(tags)
        print list_total.__len__()
        print list_gender.__len__()
        list_tag = []
        for line in list_gender:
            list_t = []
            for j in line:
                j = int(j)
                list_t.append(j)
            list_tag.append(list_t)
        print "data have read "
        return list_total, list_tag

    def load_stopword(self):
        """
        加载停用词语
        :param stopworddoc:
        :return:
        """
        stop_word = []
        return stop_word
        # with open('EN_Stopword.txt') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         word = line.replace('\n', '')
        #         if word != '':
        #             stop_word.append(word)
        # with open('ENstopwords.txt') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         word = line.replace('\n', '')
        #         if word != '':
        #             stop_word.append(word)
        #
        # return list(set(stop_word))

    # -------------------------prepare lsi svd -----------------------
    def prepare_lsi(self, doc):
        # 给训练集用的
        list_total, list_tag = self.load_data(doc)
        stop_word = self.load_stopword()
        texts = [[word for word in document.lower().split() if word not in stop_word]
                 for document in list_total]
        # train dictionary done# 抽取一个bag-of-words，将文档的token映射为id
        dictionary = corpora.Dictionary(texts)  # 生成词典 # {'a': 0, 'damaged': 1, 'gold': 3, 'fire': 2}
        # print dictionary.token2id
        # 产生文档向量，将用字符串表示的文档转换为用id和词频表示的文档向量
        corpus = [dictionary.doc2bow(text) for text in texts]
        # [[(0, 1), (6, 1)], [(0, 1), (9, 2), (10, 1)], [(0, 1), (3, 1)]]
        # 例如（9，2）这个元素代表第二篇文档中id为9的单词出现了2次
        # 用TFIDF的方法计算词频,sublinear_tf 表示学习率
        tfv = TfidfVectorizer(min_df=1, max_df=0.95, sublinear_tf=True, stop_words=stop_word)
        # 对文本中所有的用户对应的所有的评论里面的单词进行ＴＦＩＤＦ的计算，找出每个词对应的tfidf值
        X_sp = tfv.fit_transform(list_total)
        # train model done基于这些“训练文档”计算一个TF-IDF模型
        tfidf_model = models.TfidfModel(corpus)
        joblib.dump(tfidf_model, "tfidf_model.model")
        # 转化文档向量，将用词频表示的文档向量表示为一个用tf-idf值表示的文档向量
        corpus_tfidf = tfidf_model[corpus]
        # [[(1, 0.6633689723434505), (2, 0.6633689723434505)],[(7, 0.16073253746956623), (8, 0.4355066251613605)]]
        # 训练LSI模型 即将训练文档向量组成的矩阵SVD分解，并做一个秩为2的近似SVD分解
        lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=INPUT_SIZE)
        joblib.dump(dictionary, "tfidf_dictionary.dict")
        print "训练集lsi -----"
        joblib.dump(lsi_model, "tfidf_lsi.model")
        return tfidf_model, dictionary

    def train_lsi(self, doc, str_doc):
        if not (os.path.exists("tfidf_model.model")):
            print "prepare model"
            tfidf_model, dictionary = self.prepare_lsi(doc)
            list_total, list_tag = self.load_data(doc)
            stop_word = self.load_stopword()
            texts = [[word for word in document.lower().split() if word not in stop_word]
                     for document in list_total]
            corpus = [dictionary.doc2bow(text) for text in texts]
        else:
            print "use model"
            # load train valid text
            tfidf_model = joblib.load("tfidf_model.model")
            dictionary = joblib.load("tfidf_dictionary.dict")
            # load data
            list_total, list_tag = self.load_data(doc)
            stop_word = self.load_stopword()
            texts = [[word for word in document.lower().split() if word not in stop_word]
                     for document in list_total]
            corpus = [dictionary.doc2bow(text) for text in texts]
        lsi_model = joblib.load("tfidf_lsi.model")
        corpus_tfidf = tfidf_model[corpus]
        list_side = []
        corpus_lsi = lsi_model[corpus_tfidf]
        nodes = list(corpus_lsi)
        for i in range(len(nodes)):
            list_d = []
            for j in range(INPUT_SIZE):
                # print nodes[i][j]
                list_d.append(nodes[i][j][1])
            list_side.append(list_d)
        list_vec = mat(list_side)
        self.write_mat(list_vec, str_doc)
        print "lsi 矩阵构建完成----------------"
        return list_total, list_tag, list_side

    def write_mat(self, X_sp, doc_name):
        file_name = "tfidf_" + doc_name + ".npy"
        np.save(file_name, X_sp)
        print "*****************write done over *****************"
