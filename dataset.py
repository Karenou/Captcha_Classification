import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

captcha_list = list('0123456789abcdefghijklmnopqrstuvwxyz_')
captcha_length = 6


def text2vec(text):
    vector = torch.zeros((captcha_length, len(captcha_list)))
    text_len = len(text)
    if text_len > captcha_length:
        raise ValueError("验证码超过6位啦！")
    for i in range(text_len):
        vector[i,captcha_list.index(text[i])] = 1
    return vector


def vec2text(vec):
    label = torch.nn.functional.softmax(vec, dim =1)
    vec = torch.argmax(label, dim=1)
    for v in vec:
        text_list = [captcha_list[v] for v in vec]
    return ''.join(text_list)


def make_dataset(data_path):
    img_paths = glob.glob(data_path + "*.jpg")
    samples = []
    for img_path in img_paths:
        target_str = img_path.split("/")[-1].split('_')[0].lower()
        samples.append((img_path, target_str))
    return samples


class CaptchaData(Dataset):
    def __init__(self, data_path, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.samples = make_dataset(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        target_vec = text2vec(target)
        target_vec = target_vec.view(1, -1)[0]
        img = Image.open(img_path)
        img = img.resize((140,44))
        img = img.convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)
        return img, target_vec, target