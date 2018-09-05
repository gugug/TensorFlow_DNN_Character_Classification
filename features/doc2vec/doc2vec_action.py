# coding=utf-8
"""
基于doc2vec的提取文本的特征特征
"""
import codecs
import os
from sklearn.externals import joblib
import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from numpy import mat

import numpy as np

__author__ = 'gu'


class D2vAction:
    def __init__(self, base_model_dir, train_document, text_document):
        """
        初始化路径
        :param base_model_dir: 存放模型的目录
        :param train_document: 训练样本
        :param text_document: 测试样本
        :return:
        """
        self.base_model_dir = base_model_dir
        self.train_document = train_document
        self.text_document = text_document

    def train_lsi(self, doc):
        model_d2v_dm_path = os.path.join(self.base_model_dir, "model_d2v_dm.model")
        if os.path.exists(model_d2v_dm_path):
            print("已经存在模型！")
            model_dm = joblib.load(model_d2v_dm_path)
        else:
            model_dm = self.train_lsi_model(doc)
        X_doc, list_total, list_tag = self.prepare_lsi(doc)
        for i in range(10):
            # 一个用户作为一个文件去进行d2v的计算
            model_dm.train(X_doc, total_examples=model_dm.corpus_count, epochs=2)
            X_d2v = np.array([model_dm.docvecs[i] for i in range(len(list_total))])
        print(X_d2v.shape)
        list_side = X_d2v  # doc2vec 矩阵
        print(" doc2vec 矩阵构建完成----------------")
        return list_total, list_tag, list_side

    def train_lsi_model(self, doc):
        X_doc, list_total, list_tag = self.prepare_lsi(doc)
        # 训练模型
        model_dm = Doc2Vec(X_doc, dm=1, size=300, negative=5, hs=0, min_count=5, window=8, sample=1e-5, workers=4,
                           alpha=0.025, min_alpha=0.025)
        joblib.dump(model_dm, "model_d2v_dm.model")
        print("d2w模型训练完成")
        return model_dm

    def prepare_lsi(self, doc):
        # 返回文本和标签
        list_total, list_tag = self.load_data(doc)
        # 构建语料库
        X_doc = []
        TaggededDocument = gensim.models.doc2vec.TaggedDocument
        for i in range(list_total.__len__()):
            word_list = list_total[i]
            document = TaggededDocument(word_list, tags=[i])
            X_doc.append(document)
        return X_doc, list_total, list_tag

    def load_data(self, doc):
        list_name = []
        list_total = []
        list_gender = []
        # 对应标签导入词典
        f = codecs.open(doc)
        temp = f.readlines()
        f.close()

        for i in range(len(temp)):
            temp[i] = temp[i].split(" ")
            user_name = temp[i][0]
            tags = temp[i][1:6]

            query = temp[i][6:]
            query = " ".join(query).strip().replace("\n", "")

            list_total.append(query)
            list_gender.append(tags)

        list_tag = []
        for line in list_gender:
            list_t = []
            for j in line:
                j = int(j)
                list_t.append(j)
            list_tag.append(list_t)

        print("data have read ")
        return list_total, list_tag

    def get_d2v_feature(self):
        train_list_total, train_list_tag, train_list_side = self.train_lsi(self.train_document)
        print("train model done -------------------")

        text_list_total, text_list_tag, text_list_side = self.train_lsi(self.text_document)
        print("text model done  -------------------")

        TR = train_list_total.__len__()
        TE = text_list_total.__len__()
        # 将输入解释为矩阵。
        train_list_side = mat(train_list_side)
        text_list_side = mat(text_list_side)
        # train_list_tag = mat(train_list_tag, dtype=float)
        # text_list_tag = mat(text_list_tag, dtype=float)

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
        print(train_list_side)
        print(train_list_tag)
        print(text_list_side)
        print(text_list_tag)

        return train_list_side, train_list_tag, text_list_side, text_list_tag

    def write_d2v(self, filename, X_sp):
        """
        doc2vec的特征向量保存
        :param X_sp:
        :param doc_name:
        :return:
        """
        np.save(filename, X_sp)
        print("*****************write done over *****************")


def load_data_label():
    """
    加载训练数据
    :return:
    """
    base_dir = 'E:\Koo\Projects\PycharmProjects\TensorFlow_DNN_Character_Classification\data\essay_data'
    base_model_dir = ''
    d2vAction = D2vAction(base_model_dir,
                          os.path.join(base_dir, "vocab1_train.txt"),
                          os.path.join(base_dir, "vocab1_test.txt"))
    train_list_side, train_list_tag, text_list_side, text_list_tag = d2vAction.get_d2v_feature()
    str1 = "doc2vec_train_vec_dm.npy"
    str1_1 = "train_label.npy"
    str2 = "doc2vec_test_vec_dm.npy"
    str2_2 = "test_label.npy"
    d2vAction.write_d2v(str1, np.array(train_list_side))
    d2vAction.write_d2v(str1_1, np.array(train_list_tag))
    d2vAction.write_d2v(str2, np.array(text_list_side))
    d2vAction.write_d2v(str2_2, np.array(text_list_tag))


if __name__ == '__main__':
    load_data_label()
