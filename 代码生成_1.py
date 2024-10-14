"""
@飞桨/智能问答 

# 提问：
实现COO与Dense的稀疏矩阵乘，其中COO的非零元素有3个，两者的形状都为3乘3

# 参考回答：
import paddle

indices = [[0, 1, 2], [1, 2, 0]]
values = [1.0, 2.0, 3.0]
dense_shape = [3, 3]
coo = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape)
print(coo)
"""

import paddle

# 设定设备为GPU（如果有GPU可用）
paddle.device.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')

# 创建COO格式的稀疏矩阵
indices = paddle.to_tensor([[0, 1, 2], [0, 1, 2]], dtype='int64')  # 非零元素的坐标
values = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')  # 非零元素的值
shape = [3, 3]  # 稀疏矩阵的形状
coo_sparse_matrix = paddle.sparse.sparse_coo_tensor(indices, values, shape)

# 创建Dense（稠密）矩阵
dense_matrix = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype='float32')

# 使用matmul进行稀疏矩阵与稠密矩阵的乘法
result = paddle.sparse.matmul(coo_sparse_matrix, dense_matrix)

# 输出结果
print(result)
