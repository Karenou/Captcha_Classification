import torch
from torch import nn
import torchvision.models as models
from tqdm import tqdm


MAX_CAPTCHA = 6
ALL_CHAR_SET_LEN = 37


def get_model(model_name):
    if model_name == "cnn":
        return CNN()
    elif model_name == "res18":
        return RES18()
    elif model_name == "res50":
        return RES50()
    elif model_name == "mobile":
        return MOBILENET()
    elif model_name == "dense161":
        return DENSE161()
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



# ------------- attack model - simple black-box attack
def get_probs(net, x, y):
    x = x.squeeze(0)
    output = net(x)
    y = y.squeeze(0)
    y = y.type(torch.bool)
    probs = torch.nn.Softmax()(output)[:, y]
    return torch.diag(probs)

def simba_single(net, x, y, num_iters=10000, epsilon=0.2, targeted=False):
    n_dims = x.view(1, -1).size(1)
    perm = torch.randperm(n_dims)
    x = x.unsqueeze(0)
    last_prob = get_probs(net, x, y)
    for i in tqdm(range(num_iters)):
        diff = torch.zeros(n_dims)
        diff[perm[i]] = epsilon
        left_prob = get_probs(net, (x - diff.view(x.size())).clamp(0, 1), y)
        if targeted != (left_prob < last_prob):
            x = (x - diff.view(x.size())).clamp(0, 1)
            last_prob = left_prob
        else:
            right_prob = get_probs(net, (x + diff.view(x.size())).clamp(0, 1), y)
            if targeted != (right_prob < last_prob):
                x = (x + diff.view(x.size())).clamp(0, 1)
                last_prob = right_prob
    return x.squeeze()
    


# ------------ classification model
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


class MOBILENET(nn.Module):
    def __init__(self):
        super(MOBILENET, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.mobilenet_v2(pretrained=False)
        self.base.classifier = nn.Linear(self.base.last_channel, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out


class DENSE161(nn.Module):
    def __init__(self):
        super(DENSE161, self).__init__()
        self.num_cls = MAX_CAPTCHA * ALL_CHAR_SET_LEN
        self.base = models.densenet161(pretrained=False)
        self.base.classifier = nn.Linear(self.base.classifier.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        return out