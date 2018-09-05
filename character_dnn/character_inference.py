# coding=utf-8
"""
定义看前向传播的过程以及神经网络中的参数
"""

import tensorflow as tf
import math

# 神经网络相关参数
INPUT_NODE = 11  # 用户的特征维度
OUTPUT_NODE = 5  # 输出5个类别的性格
# LAYER1_NODE = 8  # 隱藏层的节点数 根据经验公式lgn
expr = 0.43 * INPUT_NODE * 5 + 0.12 * 5 * 5 + 2.54 * INPUT_NODE + 0.77 * 5 + 0.35
LAYER1_NODE = int(math.sqrt(expr) + 0.51)


def get_weight_variable(shape, regularizer):
    # 通过 tf.get_variable获取变量 和Variable 一样，在测试的时候会通过保存的模型来加载这些变量的取值。
    # 滑动平均变量重命名（影子变量），所以可以直接通过同样的变量名字取到变量本身
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        # 加入损失集合
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    """
    一层隱藏层神经网络前向传播算法
    :param input_tensor:
    :param regularizer:
    :return:
    """
    # 声明第一层神经网络的变量并完成前向传播
    with tf.variable_scope('layer1'):
        # 生成隱藏层的参数
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        # 偏置设置为0.1
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        # 使用ReLU的激活函数 去线性化
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 声明第二层神经网络的变量并完成前向传播
    with tf.variable_scope('layer2'):
        # 生成输出层的参数
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layer2 = tf.matmul(layer1, weights) + biases

    # 返回最后的前向传播的结果
    return layer2


def get_weight(shape, regularizer):
    """
    获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为’losses‘的集合中
    :param shape: 维度——对应多少个输入和多少个输出
    :param lamd: 正则化项的权重
    :return: 神经网络边上的权重
    """
    # 生成一个变量 代表权重
    var = tf.Variable(tf.random_normal(shape=shape), dtype=tf.float32)
    if regularizer != None:
        # 加入损失集合
        # 将这个权重的L2正则化损失加入名称为’losses‘的集合中
        tf.add_to_collection('losses', regularizer(var))
    # 返回一层神经网络边上的权重
    return var


def inference_nlayer(input_tensor, regularizer):
    """
    n层神经网络前向传播算法
    :param input_tensor:
    :param regularizer:
    :return:
    """
    # 定义没一层网络中的节点数
    layer_dimension = [INPUT_NODE, 100, 100, 100, OUTPUT_NODE]
    # 神经网络的层数
    n_layers = len(layer_dimension)

    # 这个变量维护前向传播时最深的层，开始时就是输入层
    cur_layer = input_tensor
    # 当前层的节点数
    in_dimension = layer_dimension[0]

    # 通过循环来生成5层全连接的神经网络结构
    for i in range(1, n_layers):
        # layer_dimension[i]为下一层的节点个数
        out_dimension = layer_dimension[i]
        # 生成当前层中权重的变量，并把这个变量的L2正则化损失加入计算图上的集合
        weight = get_weight([in_dimension, out_dimension], regularizer)
        bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

        # 使用ReLU激活函数
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
        # 进入下一层之前将下一层的节点个数更新为当前层节点个数
        in_dimension = layer_dimension[i]
    return cur_layer
