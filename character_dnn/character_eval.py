# coding=utf-8
"""
测试过程
"""
__author__ = 'gu'

import time
import tensorflow as tf
import character_inference
import numpy as np
import input_data

MOVING_AVERAGE_DECAY = 0.99  # 活动平均衰减率
MODEL_SAVE_PATH = "character_model/dbow+tfidf/"
MODEL_NAME = "character_model"
print(MODEL_SAVE_PATH)
# 加载的时间间隔。
EVAL_INTERVAL_SECS = 2

# 加载d2v 和 tfidf的数据
train_list_side, train_list_tag, text_list_side, text_list_tag = input_data.load_data_label('')

def evaluate():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, character_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.int64, name='y-input')
        validate_feed = {x: text_list_side, y_: text_list_tag}

        y = character_inference.inference(x, None)
        # y = character_inference.inference_nlayer(x, None)

        # correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        dict_acc = {}
        dict_precision = {}
        dict_recall = {}
        dict_f1 = {}
        dict_acc_lsit = {}

        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state 会根据checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # accuracy_score = sess.run(accuracy, feed_dict=validate_feed)

                    # accuracy_score = get_acc(sess,true_y, pred_y)
                    # print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))

                    # print("the input data are \n%s" % test_list_side)
                    # print("the truly answer are \n%s" % test_list_tag)
                    eval_aws = sess.run(y, feed_dict=validate_feed)
                    # print("the evaluate answer are \n%s" % eval_aws)

                    accuracy_score, acc_list = get_acc(sess, text_list_tag, eval_aws)
                    print("After %s training step(s), all validation accuracy = %g" % (global_step, accuracy_score))
                    print("After %s training step(s), 5 validation accuracy = %s" % (global_step, acc_list))

                    precision_list = get_precision(text_list_tag, eval_aws)
                    print("After %s training step(s), 5 precision = %s" % (global_step, precision_list))

                    recall_list = get_recall(text_list_tag, eval_aws)
                    print("After %s training step(s), 5 recall = %s" % (global_step, recall_list))

                    f1_list = get_f1(precision_list, recall_list)
                    print("After %s training step(s), 5 f1 = %s" % (global_step, f1_list))
                    print("==========================================")

                    if int(global_step) > 1:
                        dict_acc[global_step] = accuracy_score
                        dict_precision[global_step] = precision_list
                        dict_recall[global_step] = recall_list
                        dict_f1[global_step] = f1_list
                        dict_acc_lsit[global_step] = acc_list
                    if int(global_step) == 29001:
                        # print("================全部准确率===================")
                        # sort_dict(dict_acc)
                        print("================5个准确率===================")
                        sort_dict(dict_acc_lsit)
                        print("================5个精准率===================")
                        sort_dict(dict_precision)
                        print("================5个召回率===================")
                        sort_dict(dict_recall)
                        print("================5个f1===================")
                        sort_dict(dict_f1)
                        break

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def get_acc(sess, true_y, pred_y):
    """
    计算总的准确率和5个标签的准确率
    :param sess:
    :param true_y:
    :param pred_y:
    :return:
    """
    pred_y_ = np.where(pred_y > 0, 1, 0)
    correct_prediction = tf.equal(true_y, pred_y_)
    accuracy = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    acc_list = []
    for clazz in range(5):
        true_class1 = true_y[:, clazz]
        pred_class1 = pred_y[:, clazz]
        pred_class1_ = np.where(pred_class1 > 0, 1, 0)
        acc = 0
        for i in range(len(true_class1)):
            if true_class1[i] == pred_class1_[i]:
                acc += 1
        acc_list.append(acc * 1.0 / len(true_class1))
    return accuracy, acc_list


def get_precision(true_y, pred_y):
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
        pred_class1_ = np.where(pred_class1 > 0, 1, 0)
        precison = 0
        for i in range(len(true_class1)):
            if true_class1[i] == 1 and pred_class1_[i] == 1:
                precison += 1
        precison_list.append(precison * 1.0 / np.sum(pred_class1_))
    return precison_list


def get_recall(true_y, pred_y):
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
        pred_class1_ = np.where(pred_class1 > 0, 1, 0)
        precison = 0
        for i in range(len(true_class1)):
            if true_class1[i] == 1 and pred_class1_[i] == 1:
                precison += 1
        recall_list.append(precison * 1.0 / np.sum(true_class1))
    return recall_list


def get_f1(precison_list, recall_list):
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


def mymean(acc_list):
    acc_set = set(acc_list[1:])
    mean_acc = np.average(list(acc_set))
    print('After 20091 training steps mean_acc', mean_acc)


def sort_dict(dict):
    sorted_dict = sorted(dict.items(), key=lambda e: e[0], reverse=False)
    print(sorted_dict)
    item0 = 0
    item1 = 0
    item2 = 0
    item3 = 0
    item4 = 0
    for ke in sorted_dict:
        k = ke[1]
        # print(k)
        item0 = item0 + k[0]
        item1 = item1 + k[1]
        item2 = item2 + k[2]
        item3 = item3 + k[3]
        item4 = item4 + k[4]
    le = len(sorted_dict)
    print([item0 / le, item1 / le, item2 / le, item3 / le, item4 / le])


def main(argv=None):
    evaluate()
if __name__ == '__main__':
    tf.app.run()
