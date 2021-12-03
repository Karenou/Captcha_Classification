import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
import cv2
# from cv2 import cv2
import math
import os

from model import get_model

captcha_list = list('0123456789abcdefghijklmnopqrstuvwxyz_')
captcha_length = 6


loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

def get_accuracy(device, data_loader, test_model_name, attack_dataset_name, attack_model_name):

    net = get_model(test_model_name)
    net.to(device)
    path = './model/' # The path of pretrained models
    model = path + test_model_name
    if os.path.exists(model):
        checkpoint = torch.load(model, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    acc = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            acc += calculat_acc(outputs, labels)
            count += 1
    output = acc / count
    print('Accuracy for {} to {}_{} is: {}'.format(test_model_name, attack_model_name, attack_dataset_name, output))
    return output

# calculate RMSD
def get_RMSD(attack_model_name, attack_dataset_name):
    ori_path = './data/test/'
    attack_path = './data/attack_image/' + attack_model_name + '/' + attack_dataset_name + '/'

    ori_files = os.listdir(ori_path)
    attack_files = os.listdir(attack_path)

    i,j = 0,0
    ori_len = len(ori_files)
    attack_len = len(attack_files)

    accumulated_err = 0

    for i in range(ori_len):
        if '.png' in ori_files[i]:
            ori_img = cv2.imread(ori_path+ori_files[i])
            ori_img = cv2.resize(ori_img,(140,44))
            while j < attack_len and not ('.png' in attack_files[j]):
                j += 1
            if j < attack_len and ori_files[i] == attack_files[j]:
                att_img = cv2.imread(attack_path+attack_files[j])
                err = np.sum((ori_img.astype("float")-att_img.astype("float"))**2)
                err /= float(ori_img.shape[0] * ori_img.shape[1])
                err = math.sqrt(err)
                accumulated_err += err
                j += 1
    return accumulated_err / ori_len


def calculat_acc(output, target):

    target = target.view(-1, len(captcha_list))
    target = torch.argmax(target, dim=1)

    output = output.view(-1, len(captcha_list))
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)

    output, target = output.view(-1, captcha_length), target.view(-1, captcha_length) 
    c = 0
    for i, j in zip(target, output):
        if torch.equal(i, j):
            c += 1
    acc = c / output.size()[0] * 100
    return acc



def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image1 = torch.squeeze(image,0)
    print(image1.shape)
    return unloader(image1)
    
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  
    image = torch.squeeze(image,0)  
    print(image.shape)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def save_history(history, model_name, save_path):
#     df = pd.DataFrame.from_dict(history)
#     df.to_csv(save_path + "_loss.csv", header=True)
#
#     plt.figure(figsize=(6,4))
#     plt.plot(df["epoch"], df["train_loss"])
#     plt.xlabel("Number of Epochs")
#     plt.ylabel("Training Loss")
#     plt.title("Adversarial Attack Trained on %s" % model_name)
#     plt.savefig(save_path + "_loss.png")
#     plt.close()

