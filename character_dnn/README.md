# DNN 神经网络
运行character_train.py
修改神经网络相关参数：如下

INPUT_NODE = 11  # 用户的特征维度
OUTPUT_NODE = 5  # 输出5个类别的性格
/# LAYER1_NODE = 8  隱藏层的节点数 根据经验公式lgn
expr = 0.43 * INPUT_NODE * 5 + 0.12 * 5 * 5 + 2.54 * INPUT_NODE + 0.77 * 5 + 0.35
LAYER1_NODE = int(math.sqrt(expr) + 0.51)
