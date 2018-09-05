# -*- coding: UTF-8 -*-
''' doc2vc-svm stack for gender'''

from __future__ import division

import codecs
from sklearn import svm
from sklearn.externals import joblib

import gensim
from numpy import *
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import os
import numpy as np


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
        print(len(temp))

        for i in range(len(temp)):
            temp[i] = temp[i].split(" ")
            user_name = temp[i][0]
            tags = temp[i][1:6]

            query = temp[i][6:]
            query = " ".join(query).strip().replace("\n", "")
            list_total.append(query)
            list_gender.append(tags)

        print(list_total.__len__())
        print(list_gender.__len__())
        # 标签转化,男:0,女:1
        list_tag = []
        for line in list_gender:
            list_t = []
            for j in line:
                j = int(j)
                list_t.append(j)
            list_tag.append(list_t)

        print("data have read ")
        return list_total, list_tag

    # -------------------------prepare d2w svd -----------------------
    def prepare_lsi(self, doc):

        list_total, list_tag = self.load_data(doc)

        stop_word = []

        # 构建语料库
        X_doc = []
        TaggededDocument = gensim.models.doc2vec.TaggedDocument
        for i in range(list_total.__len__()):
            word_list = list_total[i]
            document = TaggededDocument(word_list, tags=[i])
            X_doc.append(document)

        return X_doc, list_total, list_tag

    def train_lsi_model(self, doc):

        X_doc, list_total, list_tag = self.prepare_lsi(doc)
        # 训练模型
        model_dm = Doc2Vec(X_doc, dm=0, size=300, negative=5, hs=0, min_count=1, window=30, sample=1e-5, workers=8,
                           alpha=0.04, min_alpha=0.025)
        joblib.dump(model_dm, "model_d2v_dbow.model")
        print("d2w模型训练完成")

        return model_dm

    def write_d2v(self, X_sp, doc_name):
        """
        保存doc2vec的特征向量
        :param X_sp:
        :param doc_name:
        :return:
        """
        np.save("doc2vec_" + doc_name + ".npy",X_sp)

        print("*****************write done over *****************")

    def train_lsi(self, doc, str_vec):

        if (os.path.exists("model_d2v_dbow.model")):

            # load train model
            model_dm = joblib.load("model_d2v_dbow.model")
        else:
            # load train model
            model_dm = self.train_lsi_model(doc)

        # prepare data
        X_doc, list_total, list_tag = self.prepare_lsi(doc)

        for i in range(10):
            # 一个用户作为一个文件去进行d2v的计算
            model_dm.train(X_doc, total_examples=model_dm.corpus_count, epochs=2)
            X_d2v = np.array([model_dm.docvecs[i] for i in range(len(list_total))])

        print(X_d2v.shape)

        list_side = X_d2v

        self.write_d2v(list_side, str_vec)
        print(" doc2vec 矩阵构建完成----------------")

        return list_total, list_tag, list_side

    # ------------------------my mean count------------------

    def mymean(self, list_predict_score, array_test):
        num_total = 0
        num_total = array_test.shape[0] * 5
        print("total numbers : " + str(num_total))
        return list_predict_score / (num_total)

    # ------------------------------begin to predict------------
    def predict(self):
        str1 = "train_vec_dbow"
        str2 = "test_vec_dbow"
        train_list_total, train_list_tag, train_list_side = self.train_lsi(self.train_document, str1)
        print("train model done -------------------")

        text_list_total, text_list_tag, text_list_side = self.train_lsi(self.text_document, str2)
        print("text model done  -------------------")

        TR = train_list_total.__len__()
        TE = text_list_total.__len__()
        n = 5

        train_list_side = mat(train_list_side)
        text_list_side = mat(text_list_side)

        X_train = train_list_side[:TR]
        y_train = train_list_tag[:TR]
        y_train = np.array(y_train)

        print("train shape :---------------------")
        print(X_train.shape)

        X_text = text_list_side[:TE]
        y_text = text_list_tag[:TE]
        y_text = np.array(y_text)

        print("text shape :---------------------")
        print(X_text.shape)

        # kfold折叠交叉验证
        list_myAcc = []
        true_acc = 0

        for i in range(5):
            list_train_tags = []
            list_test_tags = []
            print("第" + str(i) + "个分类器训练")

            # first build train tag
            for line in y_train:
                list_train_tags.append(line[i])

            # first build text tag
            for line in y_text:
                list_test_tags.append(line[i])

            clf = svm.SVC(probability=True)

            clf = svm.SVC(kernel='linear', probability=True)

            # 逻辑回归训练模型
            clf.fit(X_train, list_train_tags)
            # 用模型预测
            y_pred_te = clf.predict_proba(X_text)

            print(np.argmax(y_pred_te, axis=1))
            print("**" * 50)
            print(list_test_tags)

            # #获取准确的个数
            print(self.myAcc(list_test_tags, y_pred_te))
            true_acc += self.myAcc(list_test_tags, y_pred_te)

        print("true acc numbers: " + str(true_acc))

        print("d2w_dbow + 支持向量机　准确率平均值为: ")
        print(self.mymean(true_acc, X_text))


if __name__ == '__main__':
    base_dir = 'E:\\Koo\\Projects\\PycharmProjects\\TensorFlow_DNN_Character_Classification\\data\essay_data'
    user_predict = user_predict(os.path.join(base_dir, "vocab1_train.txt"), os.path.join(base_dir, "vocab1_test.txt"))
    user_predict.predict()