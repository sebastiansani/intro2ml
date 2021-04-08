import torch
import torch.nn as nn
import torchvision


class vgg_pretrained(nn.Module):
    def __init__(self):
        super(vgg_pretrained, self).__init__()
        self.vgg = torchvision.models.vgg19_bn(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, 10)
        #self.vgg.features.requires_grad_(False)

    def forward(self, x):
        return self.vgg(x)
