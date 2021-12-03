import os
import glob
import cv2
import torch
from torch.utils.data import Dataset

captcha_list = list('0123456789abcdefghijklmnopqrstuvwxyz_')
captcha_length = 6


def text2vec(text):
    vector = torch.zeros((captcha_length, len(captcha_list)))
    text_len = len(text)
    if text_len == 6:
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


def make_dataset_for_attakimages(path):
    img_names = os.listdir(path)
    dataset = []
    for img_name in img_names:
        if not ".png" in img_name:
            continue
        img = path+img_name
        label = img_name.split('.')[0].lower()
        dataset.append((img, label))
    return dataset


class CAPTCHA(Dataset):
    mode = ""

    def __init__(self, data_path, mode='train', transform=None):
        super(Dataset, self).__init__()
        self.mode = mode
        self.transform = transform
        if mode == 'eval':
            self.samples = make_dataset_for_attakimages(data_path)
        else:
            self.samples = make_dataset(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        target_vec = text2vec(target)
        target_vec = target_vec.view(1, -1)[0]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (140,44))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.mode == "attack":
            return img, target_vec, target
        else:
            return img, target_vec
