"""
@飞桨/智能问答

# 提问：

定义一个初始化函数，该函数接收一个参数，表示要初始化的模型层。如果是线性层类型，则将weight初始化为正态分布。如果是2D批归一化层类型，将weight初始化为正态分布，将bias初始化为常数0.。

# 参考回答
import paddle

def custom_initializer(layer):
    if isinstance(layer, paddle.nn.Linear):
        paddle.nn.initializer.Normal(mean=0.0, std=0.01)(layer.weight)
    elif isinstance(layer, paddle.nn.BatchNorm2D):
        paddle.nn.initializer.Normal(mean=1.0, std=0.02)(layer.weight)
        paddle.nn.initializer.Constant(value=0.0)(layer.bias)
    else:
        print(f"No special initialization for layer type {type(layer)}.")

if __name__ == "__main__":
    linear_layer = paddle.nn.Linear(in_features=10, out_features=20)
    bn_layer = paddle.nn.BatchNorm2D(num_features=32)

    custom_initializer(linear_layer)
    custom_initializer(bn_layer)

    print("Linear layer weight initialized:", linear_layer.weight.numpy())
    print("BatchNorm2D layer weight initialized:", bn_layer.weight.numpy())
    print("BatchNorm2D layer bias initialized:", bn_layer.bias.numpy())
"""

import paddle
import paddle.nn as nn

def custom_init(layer):
    """
    自定义初始化函数
    
    Args:
        layer (nn.Layer): 要初始化的模型层
    """
    if isinstance(layer, nn.Linear):
        # 如果是线性层，将weight初始化为正态分布，均值为0，标准差为0.01
        nn.initializer.Normal(mean=0.0, std=0.01)(layer.weight)
        # 偏置初始化为0
        nn.initializer.Constant(value=0.0)(layer.bias)
    elif isinstance(layer, nn.BatchNorm2D):
        # 如果是2D批归一化层，将weight初始化为正态分布，均值为1，标准差为0.02（通常BN层的weight初始化为1）
        nn.initializer.Normal(mean=1.0, std=0.02)(layer.weight)
        # 偏置初始化为0
        nn.initializer.Constant(value=0.0)(layer.bias)

# 示例使用
linear_layer = nn.Linear(10, 20)
custom_init(linear_layer)

bn_layer = nn.BatchNorm2D(num_features=64)
custom_init(bn_layer)

# 注意：对于BatchNorm2D层，其weight的初始化通常设为1，而不是从0开始的正态分布，
# 因为BatchNorm层在训练初期需要稳定的输入方差来加速训练过程。
