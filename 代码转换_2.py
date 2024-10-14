import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear

class MyNet(nn.Module):
    test = "str"

    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        self._conv = torch.nn.Conv2d(4, 6, (3, 3))
        self._pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self._fc1 = torch.nn.Linear(6 * 25 * 25, 120)  # 假设输入图像为28x28，通过卷积和池化后尺寸变为25x25
        self._fc2 = nn.Linear(120, out_features=84)
        self._fc3 = Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self._conv(x)
        x = self._pool(x)

        x = self._fc1(torch.flatten(x, 1))
        x = self._fc2(x)
        x = self._fc3(x)
        y = torch.add(x, x)
        return y

net = MyNet()
sgd = optim.SGD(net.parameters(), lr=0.01)
lr = optim.lr_scheduler.MultiStepLR(sgd, milestones=[2, 4, 6], gamma=0.8)

for i in range(10):
    x = torch.rand(8, 4, 28, 28)
    out = net(x).sum()

    sgd.zero_grad()
    out.backward()
    sgd.step()
