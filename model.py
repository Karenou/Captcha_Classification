import torch
from torch import nn
import torchvision.models as models


def get_model(opt):
    if opt.model == "cnn":
        return CNN()
    elif opt.model == "vgg16":
        return VGG16()
    elif opt.model == "res18":
        return RES18()
    elif opt.model == "res50":
        return RES50()
    elif opt.model == "res101":
        return RES101()
    elif opt.model == "mobile":
        return MOBILENETV2()
    elif opt.model == "incep1":
        return INCEPTIONV1()
    elif opt.model == "incep3":
        return INCEPTIONV3()
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )
 
        self.fc1 = nn.Sequential(
            nn.Linear(4*16*128, 1024),
            nn.Dropout(0.2),  
            nn.ReLU()
        )

        self.fc2 = nn.Linear(1024, MAX_CAPTCHA * ALL_CHAR_SET_LEN) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


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


class MOBILENETV2(nn.Module):
    def __init__(self):
        super(MOBILENETV2, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.mobilenet_v2(pretrained=False)
        self.base.classifier = nn.Linear(self.base.last_channel, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out

class INCEPTIONV1(nn.Module):
    def __init__(self):
        super(INCEPTIONV1, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.googlenet(pretrained=False)
        self.base.classifier = nn.Linear(1024, self.num_cls)
    def forward(self, x):
        out = self.base(x).logits
        return out

class INCEPTIONV3(nn.Module):
    def __init__(self):
        super(INCEPTIONV3, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.inception_v3(pretrained=False)
        self.base.classifier = nn.Linear(2048, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out
