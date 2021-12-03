import csv

import numpy
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification import attack_dataset_list, model_list, device
from dataset import CAPTCHA
from util import get_accuracy, get_RMSD


def load_datasets(attack_model_name, dataset):
    transform = transforms.Compose([transforms.ToTensor()])
    path = './data/attack_image/' + attack_model_name + '/' + dataset + '/'
    # print(path, os.listdir(path))
    data = CAPTCHA(path, mode='eval', transform=transform)
    data_loader = DataLoader(data, batch_size=1)
    return data_loader




def test_attack_images():
    f = open('result.csv','w')
    writer = csv.writer(f)

    for dataset in tqdm(attack_dataset_list):
        for test_model in model_list:
            line_acc = []
            for attack_model in model_list:
                attack_dataset = load_datasets(attack_model, dataset)
                result = get_accuracy(device, attack_dataset, test_model, dataset, attack_model)
                line_acc.append(result)
            writer.writerow(numpy.array(line_acc))
        writer.writerow("")
    f.close()

def test_original():
    transform = transforms.Compose([transforms.ToTensor()])
    path = './data/test/'
    data = CAPTCHA(path, mode='test', transform=transform)
    data_loader = DataLoader(data, batch_size=128, shuffle=False, drop_last=True)
    for model in tqdm(model_list):
        get_accuracy(device, data_loader, model, "raw_text", "")


def test_RMSD():
    f = open('RMSD.csv', 'w')
    writer = csv.writer(f)

    for dataset in attack_dataset_list:
        for attack_model in model_list:
            tmp = get_RMSD(attack_model, dataset)
            line = [attack_model, tmp]
            writer.writerow(line)
            # print(line)
        writer.writerow('')
    f.close()

if __name__ == '__main__':
    test_original()
    test_attack_images()
    test_RMSD()