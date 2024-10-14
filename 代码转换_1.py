import torch

x = torch.tensor([[1, 2], [4, 5]])
mat1 = torch.tensor([[1, 2], [4, 5]])
mat2 = torch.tensor([[1, 2], [4, 5]])
result1 = torch.addmm(x, mat1, mat2)

result2 = torch.addmm(x, mat1, mat2, beta=0.6, alpha=0.7)

result3 = torch.addmm(x, mat1=mat1, mat2=mat2, beta=0.6, alpha=0.7)

result4 = torch.addmm(input=x, mat1=mat1, mat2=mat2, beta=0.6, alpha=0.7)

result5 = torch.addmm(alpha=0.7, input=x, beta=0.6, mat1=mat1, mat2=mat2)
