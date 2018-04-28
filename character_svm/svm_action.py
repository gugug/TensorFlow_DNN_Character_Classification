# -*- coding: UTF-8 -*-
"""
svm tfidf_d2v_dm_dbow_textmind_emotion for character
calculate their acc precision recall f1 value
"""

from __future__ import division

from sklearn import svm
from numpy import *
import numpy as np
import os
from features.crawl_textmind_data import input_textmind_data
from utils import logger

LOG = logger.get_logger()


class SVMCharacterPredict:
    def myAcc(self, y_true, y_pred):
        """
        准确值计算
        :param y_true:
        :param y_pred:
        :return:
        """
        true_num = 0
        # for i in range(y_true.__len__()):
        #     # print y_true[i]
        for i in range(y_pred.__len__()):
            if y_true[i] == y_pred[i]:
                true_num += 1
        return true_num

    def mymean(self, list_predict_score, array_test):
        """
        my mean count
        :param list_predict_score:
        :param array_test:
        :return:
        """
        num_total = 0
        num_total = array_test.shape[0] * 5
        # print "total numbers : " + str(num_total)
        return list_predict_score / (num_total)

    def train_eval(self, X_train, y_train, X_text, y_text):
        """
        输入矩阵 训练模型并计算准确率
        :param X_text:
        :param X_train:
        :param y_text:
        :param y_train:
        :return:
        """
        pred_y = []
        true_acc = 0
        for i in range(5):
            list_train_tags = []
            list_test_tags = []
            # # print "第" + str(i) + "个分类器训练"
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
            # # print np.argmax(y_pred_te, axis=1)
            # # print "**" * 50
            # # print list_test_tags
            # #获取准确的个数
            # # print self.myAcc(list_test_tags, y_pred_te)

            # 最大数的索引
            y_pred = np.argmax(y_pred_te, axis=1)
            true_acc += self.myAcc(list_test_tags, y_pred)
            pred_y.append(y_pred)

        # print "true acc numbers: " + str(true_acc)
        pred_y_ = map(list, zip(*pred_y))
        pred_y_ = mat(pred_y_)
        return self.mymean(true_acc, X_text), pred_y_

    def predict_by_textmind(self):
        """
        svm 文心特征
        :return:
        """
        X_train, Y_train, X_test, Y_test = input_textmind_data.load_textmind_data_label_with_normalization(
            '../crawl_textmind_data')
        mymean, pred_y = self.train_eval(X_train, Y_train, X_test, Y_test)
        print "textmind+支持向量机　准确率平均值为: " + str(mymean)
        # LOG.info("textmind+支持向量机　准确率平均值为: " + str(mymean))

        acc_list = self.get_acc(Y_test, pred_y)
        print("After training step(s), 5 validation accuracy = %s" % acc_list)
        precision_list = self.get_precision(Y_test, pred_y)
        print("After training step(s), 5 precision = %s" % precision_list)
        recall_list = self.get_recall(Y_test, pred_y)
        print("After training step(s), 5 recall = %s" % recall_list)
        f1_list = self.get_f1(precision_list, recall_list)
        print("After training step(s), 5 f1 = %s" % f1_list)
        print("==========================================")
        return X_train, Y_train, X_test, Y_test

    def predict_by_d2v_dm(self):
        """
        d2v_dm 训练
        :return:
        """
        base_model_dir = ''
        train_vec_filename = os.path.join(base_model_dir, "doc2vec_train_vec_dm.npy")
        train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
        test_vec_filename = os.path.join(base_model_dir, 'doc2vec_test_vec_dm.npy')
        test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

        X_train, Y_train, X_test, Y_test = self.load_arr(test_label_filename, test_vec_filename, train_label_filename,
                                                         train_vec_filename)
        mymean, pred_y = self.train_eval(X_train, Y_train, X_test, Y_test)
        # print "d2v_dm+支持向量机　准确率平均值为: " + str(mymean)
        LOG.info("d2v_dm+支持向量机　准确率平均值为: " + str(mymean))
        return X_train, Y_train, X_test, Y_test

    def predict_by_d2v_dbow(self):
        """
        d2v_dbow 训练
        :return:
        """
        base_model_dir = ''
        train_vec_filename = os.path.join(base_model_dir, "doc2vec_train_vec_dbow.npy")
        train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
        test_vec_filename = os.path.join(base_model_dir, 'doc2vec_test_vec_dbow.npy')
        test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

        X_train, Y_train, X_test, Y_test = self.load_arr(test_label_filename, test_vec_filename, train_label_filename,
                                                         train_vec_filename)
        mymean, pred_y = self.train_eval(X_train, Y_train, X_test, Y_test)
        print "d2v_dbow+支持向量机　准确率平均值为: " + str(mymean)
        # LOG.info("d2v_dbow+支持向量机　准确率平均值为: " + str(mymean))

        acc_list = self.get_acc(Y_test, pred_y)
        print("After training step(s), 5 validation accuracy = %s" % acc_list)
        precision_list = self.get_precision(Y_test, pred_y)
        print("After training step(s), 5 precision = %s" % precision_list)
        recall_list = self.get_recall(Y_test, pred_y)
        print("After training step(s), 5 recall = %s" % recall_list)
        f1_list = self.get_f1(precision_list, recall_list)
        print("After training step(s), 5 f1 = %s" % f1_list)
        print("==========================================")

        return X_train, Y_train, X_test, Y_test

    def predict_by_tfidf(self):
        """
        tfidf 训练
        :return:
        """
        base_model_dir = ''
        train_vec_filename = os.path.join(base_model_dir, "tfidf_train_vec_tfidf.npy")
        train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
        test_vec_filename = os.path.join(base_model_dir, 'tfidf_test_vec_tfidf.npy')
        test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

        X_train, Y_train, X_test, Y_test = self.load_arr(test_label_filename, test_vec_filename, train_label_filename,
                                                         train_vec_filename)
        mymean, pred_y = self.train_eval(X_train, Y_train, X_test, Y_test)
        print "tfidf+支持向量机　准确率平均值为: " + str(mymean)
        # LOG.info("tfidf + 停用词 +支持向量机　准确率平均值为: " + str(mymean))

        acc_list = self.get_acc(Y_test, pred_y)
        print("After training step(s), 5 validation accuracy = %s" % acc_list)
        precision_list = self.get_precision(Y_test, pred_y)
        print("After training step(s), 5 precision = %s" % precision_list)
        recall_list = self.get_recall(Y_test, pred_y)
        print("After training step(s), 5 recall = %s" % recall_list)
        f1_list = self.get_f1(precision_list, recall_list)
        print("After training step(s), 5 f1 = %s" % f1_list)
        print("==========================================")

        return X_train, Y_train, X_test, Y_test

    def predict_by_tfidf_stopword(self):
        """
        tfidf 训练
        :return:
        """
        base_model_dir = ''
        train_vec_filename = os.path.join(base_model_dir, "tfidf_train_vec_tfidf_stopword.npy")
        train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
        test_vec_filename = os.path.join(base_model_dir, 'tfidf_test_vec_tfidf_stopword.npy')
        test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

        X_train, Y_train, X_test, Y_test = self.load_arr(test_label_filename, test_vec_filename, train_label_filename,
                                                         train_vec_filename)
        mymean, pred_y = self.train_eval(X_train, Y_train, X_test, Y_test)
        print "tfidf+stopword+支持向量机　准确率平均值为: " + str(mymean)
        # LOG.info("tfidf+支持向量机　准确率平均值为: " + str(mymean))

        acc_list = self.get_acc(Y_test, pred_y)
        print("After training step(s), 5 validation accuracy = %s" % acc_list)
        precision_list = self.get_precision(Y_test, pred_y)
        print("After training step(s), 5 precision = %s" % precision_list)
        recall_list = self.get_recall(Y_test, pred_y)
        print("After training step(s), 5 recall = %s" % recall_list)
        f1_list = self.get_f1(precision_list, recall_list)
        print("After training step(s), 5 f1 = %s" % f1_list)
        print("==========================================")

        return X_train, Y_train, X_test, Y_test

    def load_arr(self, test_label_filename, test_vec_filename, train_label_filename, train_vec_filename):
        X_train = np.load(train_vec_filename)
        # # print('X_train', X_train.shape)
        Y_train = np.load(train_label_filename)
        # # print('Y_train', Y_train.shape)
        X_test = np.load(test_vec_filename)
        # # print('X_test', X_test.shape)
        Y_test = np.load(test_label_filename)
        # # print('Y_test', Y_test.shape)
        return X_train, Y_train, X_test, Y_test

    def predict_by_emotion(self):
        """
        情感特征
        :return:
        """
        from features.emotion_lexicon import data_helper

        X_train, Y_train, X_test, Y_test = data_helper.load_emotion_data_label('../Emotion_Lexicon')
        mymean, pred_y = self.train_eval(X_train, Y_train, X_test, Y_test)
        # print "情感特征+支持向量机　准确率平均值为: " + str(mymean)
        LOG.info("情感特征+支持向量机　准确率平均值为: " + str(mymean))
        return X_train, Y_train, X_test, Y_test

    def predict_by_combine(self):
        """
        组合特征训练
        :return:
        """
        from character_dnn import input_data

        base_model_dir = ''
        train_vec_filename = os.path.join(base_model_dir, "tfidf_train_vec_tfidf.npy")
        train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
        test_vec_filename = os.path.join(base_model_dir, 'tfidf_test_vec_tfidf.npy')
        test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

        X_train, Y_train, X_test, Y_test = self.load_arr(test_label_filename, test_vec_filename, train_label_filename,
                                                         train_vec_filename)
        train_vec_filename = os.path.join(base_model_dir, "doc2vec_train_vec_dbow.npy")
        train_label_filename = os.path.join(base_model_dir, 'doc2vec_train_label_dm.npy')
        test_vec_filename = os.path.join(base_model_dir, 'doc2vec_test_vec_dbow.npy')
        test_label_filename = os.path.join(base_model_dir, 'doc2vec_test_label_dm.npy')

        X1_train, Y1_train, X1_test, Y1_test = self.load_arr(test_label_filename, test_vec_filename,
                                                             train_label_filename,
                                                             train_vec_filename)

        train_list_side, text_list_side = input_data.load_data_label_combine(X_train, X_test, X1_train, X1_test)
        mymean, pred_y = self.train_eval(train_list_side, Y_train, text_list_side, Y_test)
        # print "综合特征+支持向量机　准确率平均值为: " + str(mymean)
        LOG.info("tfidf+dbow+综合特征+支持向量机　准确率平均值为: " + str(mymean))

        acc_list = self.get_acc(Y_test, pred_y)
        print("After training step(s), 5 validation accuracy = %s" % acc_list)
        precision_list = self.get_precision(Y_test, pred_y)
        print("After training step(s), 5 precision = %s" % precision_list)
        recall_list = self.get_recall(Y_test, pred_y)
        print("After training step(s), 5 recall = %s" % recall_list)
        f1_list = self.get_f1(precision_list, recall_list)
        print("After training step(s), 5 f1 = %s" % f1_list)
        print("==========================================")

    def predict_by_combine_two(self, fun1, fun2, fun1name, fun2name):
        from character_dnn import input_data

        X_train, Y_train, X_test, Y_test = fun1
        X1_train, Y1_train, X1_test, Y1_test = fun2
        train_list_side, text_list_side = input_data.load_data_label_combine(X_train, X_test, X1_train, X1_test)
        mymean, pred_y = self.train_eval(train_list_side, Y_train, text_list_side, Y_test)
        print "综合特征+支持向量机　准确率平均值为: " + str(mymean)
        LOG.info(fun1name + " + " + fun2name + " 综合特征+支持向量机　准确率平均值为: " + str(mymean))

        acc_list = self.get_acc(Y_test, pred_y)
        print("After training step(s), 5 validation accuracy = %s" % acc_list)
        precision_list = self.get_precision(Y_test, pred_y)
        print("After training step(s), 5 precision = %s" % precision_list)
        recall_list = self.get_recall(Y_test, pred_y)
        print("After training step(s), 5 recall = %s" % recall_list)
        f1_list = self.get_f1(precision_list, recall_list)
        print("After training step(s), 5 f1 = %s" % f1_list)
        print("==========================================")

    def predict_by_combine_three(self):
        from character import input_data

        X_train, Y_train, X_test, Y_test = self.predict_by_tfidf()
        X1_train, Y1_train, X1_test, Y1_test = self.predict_by_d2v_dbow()
        X2_train, Y2_train, X2_test, Y2_test = self.predict_by_emotion()
        X3_train, X3_test = input_data.load_data_label_combine(X_train, X_test, X1_train, X1_test)
        train_list_side, text_list_side = input_data.load_data_label_combine(X3_train, X3_test, X2_train, X2_test)
        mymean, pred_y = self.train_eval(train_list_side, Y_train, text_list_side, Y_test)
        # print "综合特征+支持向量机　准确率平均值为: " + str(mymean)
        LOG.info(" tiidf + d2v_dbow + emotion 综合特征+支持向量机　准确率平均值为: " + str(mymean))

    def get_acc(self, true_y, pred_y):
        """
        计算总的准确率和5个标签的准确率
        :param sess:
        :param true_y:
        :param pred_y:
        :return:
        """
        acc_list = []
        for clazz in range(5):
            true_class1 = true_y[:, clazz]
            pred_class1 = pred_y[:, clazz]
            acc = 0
            for i in range(len(true_class1)):
                if true_class1[i] == pred_class1[i]:
                    acc += 1
            acc_list.append(acc * 1.0 / len(true_class1))
        return acc_list

    def get_precision(self, true_y, pred_y):
        """
        返回五个标签的精确率
        :param true_y:
        :param pred_y:
        :return:
        """
        precison_list = []
        for clazz in range(5):
            true_class1 = true_y[:, clazz]
            pred_class1 = pred_y[:, clazz]
            precison = 0
            for i in range(len(true_class1)):
                if true_class1[i] == 1 and pred_class1[i] == 1:
                    precison += 1
            precison_list.append(precison * 1.0 / np.sum(pred_class1))
        return precison_list

    def get_recall(self, true_y, pred_y):
        """
        返回5个标签的召回率
        :param true_y:
        :param pred_y:
        :return:
        """
        recall_list = []
        for clazz in range(5):
            true_class1 = true_y[:, clazz]
            pred_class1 = pred_y[:, clazz]
            precison = 0
            for i in range(len(true_class1)):
                if true_class1[i] == 1 and pred_class1[i] == 1:
                    precison += 1
            recall_list.append(precison * 1.0 / np.sum(true_class1))
        return recall_list

    def get_f1(self, precison_list, recall_list):
        """
        返回5个标签的f1值
        :param precison:
        :param recall:
        :return:
        """
        f1_list = []
        for i in range(5):
            precison = precison_list[i]
            recall = recall_list[i]
            f1_list.append((2 * precison * recall) / (precison + recall))
        return f1_list




if __name__ == '__main__':
    user_predict = SVMCharacterPredict()
    # user_predict.predict_by_combine()
    # for _ in range(2):
    #     user_predict.predict_by_combine_three()

    # # 训练10次
    for _ in range(10):
        LOG.info("=========开始第" + str(_ + 1) + "轮训练组合===========")
        # fun2 = user_predict.predict_by_textmind()
        # fun3 = user_predict.predict_by_d2v_dbow()
        # fun5 = user_predict.predict_by_tfidf()
        # user_predict.predict_by_tfidf_stopword()
        f2name = 'textmind'
        f3name = 'dbow'
        f5name = 'tfidf'
        # user_predict.predict_by_combine_two(fun2, fun3, f2name, f3name)
        # user_predict.predict_by_combine_two(fun2, fun5, f2name, f5name)
        # user_predict.predict_by_combine_two(fun3, fun5, f3name, f5name)
        user_predict.predict_by_combine()
