import torch
from torch import nn
import torchvision.models as models


def get_model(opt):
    if opt.model == "vgg16":
        return VGG16()
    elif opt.model == "res18":
        return RES18()
    elif opt.model == "res50":
        return RES50()
    elif opt.model == "res101":
        return RES101()
    elif opt.model == "mobile":
        return MOBILENET()
    elif opt.model == "googlenet":
        return GOOGLENET()
    else:
        raise ValueError("please input model name")


def get_optimizer(opt, net):
    if opt.optimizer == "adam":
        return torch.optim.Adam(net.parameters(), lr=opt.lr)
    elif opt.optimizer == "sgd":
        return torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    elif opt.optimizer == "rmsprop":
        return torch.optim.Rprop(net.parameters(), lr=opt.lr)
    else:
        raise ValueError("please input model name")

    

MAX_CAPTCHA = 6
ALL_CHAR_SET_LEN = 37

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.vgg16(pretrained=False)
        self.base.classifier[-1] = nn.Linear(4096, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out
        

class RES18(nn.Module):
    def __init__(self):
        super(RES18, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.resnet18(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out


class RES50(nn.Module):
    def __init__(self):
        super(RES50, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.resnet50(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class RES101(nn.Module):
    def __init__(self):
        super(RES101, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.resnet101(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out


class MOBILENET(nn.Module):
    def __init__(self):
        super(MOBILENET, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.mobilenet_v2(pretrained=False)
        self.base.classifier = nn.Linear(self.base.last_channel, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out


class GOOGLENET(nn.Module):
    def __init__(self):
        super(GOOGLENET, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.googlenet(pretrained=False)
        self.base.classifier = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out
