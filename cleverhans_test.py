from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision.models as models
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from torchvision import utils,transforms
import matplotlib as plt
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from torchvision.utils import save_image


FLAGS = flags.FLAGS

captcha_list = list('0123456789abcdefghijklmnopqrstuvwxyz_')
captcha_length = 6

# 验证码文本转为向量
def text2vec(text):
    vector = torch.zeros((captcha_length, len(captcha_list)))
    text_len = len(text)
    if text_len > captcha_length:
        raise ValueError("验证码超过6位啦！")
    for i in range(text_len):
        vector[i,captcha_list.index(text[i])] = 1
    # print(text)
    return vector

# 验证码向量转为文本
def vec2text(vec):
    label = torch.nn.functional.softmax(vec, dim =1)
    vec = torch.argmax(label, dim=1)
    for v in vec:
        text_list = [captcha_list[v] for v in vec]
    return ''.join(text_list)

def make_dataset(data_path):
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = data_path+img_name
        target_str = img_name.split('_')[0].lower()
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
        target = text2vec(target)
        target = target.view(1, -1)[0]
        img = Image.open(img_path)
        img = img.resize((140,44))
        img = img.convert('RGB') # img转成向量
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class RES18(nn.Module):
    def __init__(self):
        super(RES18, self).__init__()
        self.num_cls = 6*37
        self.base = models.resnet18(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, self.num_cls)
    def forward(self, x):
        out = self.base(x)
        #print("the output shape is:")
        #print(out.shape)
        return out

loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image1 = torch.squeeze(image,0)
    print(image1.shape)
    return unloader(image1)
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = torch.squeeze(image,0)  # remove the fake batch dimension
    print(image.shape)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def main(_):
    # Load training and test data
    #data = ld_cifar10()
    # 数据准备
    transform = transforms.Compose([transforms.ToTensor()])  # 不做数据增强和标准化了
    train_dataset = CaptchaData('trainfile/', transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True, drop_last=True)

    test_data = CaptchaData('testfile/', transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=15, num_workers=0, shuffle=True, drop_last=True)

    # Instantiate model, loss, and optimizer for training
    net = RES18()
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train vanilla model
    net.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in train_data_loader:
            x, y = x.to(device), y.to(device)
            print(x.shape)
            print(y.shape)
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )

    # Evaluate on clean and adversarial data
    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in test_data_loader:
        x, y = x.to(device), y.to(device)
        print(x.shape)
        print(y.shape)
        x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
        x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)

        """
        xs = x.permute(0,2,3,1)
        x_pgds = x_pgd.permute(0,2,3,1)
        x_fgms = x_fgm.permute(0,2,3,1)
        """
        img1 = x[0]
        print(img1.shape)
        save_image(img1, 'img1.png')
        img2 = x_pgd[0]
        save_image(img2, 'img2.png')
        img3 = x_fgm[0]
        save_image(img3, 'img3.png')

        y_pred = net(x)  # model prediction on clean examples
        y_pred_fgm = net(x_fgm) # model prediction on FGM adversarial examples
        y_pred_pgd = net(x_pgd) # model prediction on PGD adversarial examples
        print("y pre shape:")
        print(y_pred.shape)
        print("pgd pre shape:")
        print(y_pred_pgd)
        print("y pre shale: ")
        print(y_pred_fgm)
        report.nb_test += y.size(0)
        report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_pgd += y_pred_pgd.eq(y).sum().item()
        report.correct += y_pred.eq(y).sum().item()

    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            report.correct_fgm / report.nb_test * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report.correct_pgd / report.nb_test * 100.0
        )
    )


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )

    app.run(main)