import torch.nn as nn
from collections import OrderedDict

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # AlexNet 包含 5个卷积层
        self.features = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ('s2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('c3', nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False)),
            ('relu3', nn.ReLU(inplace=True)),
            ('s4', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('c5', nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False)),
            ('relu5', nn.ReLU(inplace=True)),
            ('c6', nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False)),
            ('relu6', nn.ReLU(inplace=True)),
            ('c7', nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ('relu7', nn.ReLU(inplace=True)),
            ('s8', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]))
        
        # 包含 3个全连接层
        self.classifier = nn.Sequential(OrderedDict([
            ('f9', nn.Linear(256 * 6 * 6, 4096)),
            ('relu9', nn.ReLU(inplace=True)),
            ('f10', nn.Linear(4096, 4096)),
            ('relu10', nn.ReLU(inplace=True)),
            ('f11', nn.Linear(4096, num_classes)),
        ]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x