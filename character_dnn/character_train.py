# coding=utf-8
"""
定义了神经网络的训练过程
"""

import tensorflow as tf
import character_inference
import os
import input_data

# 1. 定义神经网络结构相关的参数。
BATCH_SIZE = 50  # 一个训练batch中的训练数据个数，数字越小，训练过程越接近随机梯度下降
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化在损失函数的系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 活动平均衰减率
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "character_model"

# 加载d2v 和 tfidf的数据
train_list_side, train_list_tag, text_list_side, text_list_tag = input_data.load_data_label('')
TRAIN_NUM_EXAMPLES = DATASET_SIZE = len(train_list_side)  # 训练数据的总数

# 2. 定义训练过程。
def train():
    # 定义输入输出placeholder。
    x = tf.placeholder(tf.float32, [None, character_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, name='y-input')
    # L2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算在当前参数下神经网络前向传播的结果
    y = character_inference.inference(x, regularizer)
    # y = character_inference.inference_nlayer(x,regularizer)
    # 定义存储训练轮数的便利那个。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量
    global_step = tf.Variable(0, trainable=False)

    # ///////////////====定义损失函数、学习率、滑动平均操作以及训练过程。=====//////////////
    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵作为刻画预测值和真实值之间茶军的损失函数
    """
    // 参考损失函数的计算 http://blog.csdn.net/u013250416/article/details/78230464
    sigmoid_cross_entropy_with_logits  应用于多标签或者二分类
    """
    # 多目标损失函数
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, targets=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 指数衰减设置学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        DATASET_SIZE / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    # 优化损失函数,在minimize中传入global_step将自动更新global_step,从而更新学习率
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时既需要通过反向传播来更新神经网络的参数，又要更新每一个参数的滑动平均值。
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAINING_STEPS):

            # # 每次选取batch_size样本进行训练
            # start = (i * BATCH_SIZE) % DATASET_SIZE
            # end = min(start + BATCH_SIZE, DATASET_SIZE)
            # _, loss_value, step = sess.run([train_op, loss, global_step],
            #                                feed_dict={x: train_list_side[start:end],
            #                                           y_: train_list_tag[start:end]})

            # 每次选取all_size样本进行训练
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: train_list_side,
                                                      y_: train_list_tag})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    train()
if __name__ == '__main__':
    tf.app.run()
