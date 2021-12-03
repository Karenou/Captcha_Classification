import math

import numpy
import numpy as np
import torch
import cv2
from torch import nn
import torchvision.transforms as transforms
from torch.nn.functional import normalize
from tqdm import tqdm
import csv
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os

from dataset import text2vec, CAPTCHA
from model import *
from util import get_RMSD, calculat_acc, get_accuracy




captcha_list = list('0123456789abcdefghijklmnopqrstuvwxyz_')
captcha_length = 6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# The following are the batch testing for the accuracy of each kinds of attack models.
# Datasets
noise_eps_1 = "noise_eps_0.1"
noise_eps_2 = "noise_eps_0.2"
simba_eps_1 = "simba_eps_0.1"
simba_eps_2 = "simba_eps_0.2"
fgps_eps_1 = "fgsm_eps_0.1"
fgsm_eps_2 = "fgsm_eps_0.2"
pgd_eps_1 = "pgd_eps_0.1"
pgd_eps_2 = "pgd_eps_0.2"
sparse_eps_20 = "sparse_eps_20.0"
sparse_eps_30 = "sparse_eps_30.0"
attack_dataset_list = [noise_eps_1, noise_eps_2, simba_eps_1, simba_eps_2, fgps_eps_1,
                fgsm_eps_2, pgd_eps_1, pgd_eps_2, sparse_eps_20, sparse_eps_30]

# Models
cnn = "cnn"
res50 = "res50"
res18 = "res18"
mobile = "mobile"
dense161 = "dense161"
model_list = [cnn, res18, res50, mobile, dense161]



def train(epoch_nums, model_name):
    net = get_model(model_name)
    net.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CAPTCHA('./data/train/', mode="train", transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, drop_last=True)

    loss_function = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    model_path = './model/{}.pth'.format(model_name)
    i = 1
    for epoch in tqdm(range(epoch_nums)):
        running_loss = 0.0
        net.train()
        for data in tqdm(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 0:
                torch.save({'model_state_dict':net.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),}, model_path)
            i += 1
        if epoch % 5 == 4:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9




### model_list = [selfcnn, restnet18, restnet50, mobilenet, densenet161] used for training
if __name__ == '__main__':
    # train classification model
    train(10, cnn)
    train(10, res18)
    train(10, res50)
    train(10, mobile)
    train(10, dense161)
