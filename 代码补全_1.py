import paddle

def calculate_softmax(x, dim):
    return paddle.nn.functional.softmax(x, axis=dim)
    