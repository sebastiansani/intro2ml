import torch
import torch.nn as nn
import torchvision


num_classes = 10


class vgg_pretrained(nn.Module):
    def __init__(self):
        super(vgg_pretrained, self).__init__()
        self.vgg = torchvision.models.vgg19_bn(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        # self.vgg.features.requires_grad_(False)

    def forward(self, x):
        return self.vgg(x)


class resnet_pretrained(nn.Module):
    def __init__(self):
        super(resnet_pretrained, self).__init__()
        self.res = torchvision.models.resnet152(pretrained=True)
        self.res.fc = nn.Linear(4*512, num_classes)

    def forward(self, x):
        return self.res(x)
