import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)
from torchvision.utils import save_image

from model import *
from dataset import CaptchaData
from util import save_history


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
parser.add_argument("--model", type=str, default="res50", help="Pre-trained model.")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for model during training.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the optimizer during training")
parser.add_argument("--eps", type=float, default=0.1, help="Total epsilon for FGM and PGD attacks.")
parser.add_argument("--pgd_nb_iter", type=int, default=30, help="Number of iteration for PGD update.")
parser.add_argument("--mode", type=str, default="train", help="Train or test mode.")
opt = parser.parse_args()
print(opt)

def cross_entropy_loss_one_hot(input, labels, batch_size):
    log_prob = torch.nn.functional.log_softmax(input, dim=1)
    loss = -torch.sum(log_prob * labels) / batch_size
    return loss


if __name__ == "__main__":
    # Load training and test data
    transform = transforms.Compose([transforms.ToTensor()])  
    train_dataset = CaptchaData('./data/train/', transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    test_data = CaptchaData('./data/test/', transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=2, shuffle=True, drop_last=True)

    # Instantiate model, loss, and optimizer for training
    net = get_model(opt)

    print("GPU available: ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()

    # loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = get_optimizer(opt, net)

    # Train vanilla model
    if opt.mode == "train":
        net.train()
        print("start model training")
        history = {"epoch": [], "train_loss": []}
        for epoch in range(1, opt.n_epochs + 1):
            train_loss = 0.0
            step = 1
            for x, y, _ in train_data_loader:
                x, y = x.to(device),  y.to(device, dtype=torch.int64)
                optimizer.zero_grad()
                loss = cross_entropy_loss_one_hot(net(x), y, opt.batch_size)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                step += 1
            
            print("epoch: {}/{}, train loss: {:.3f}".format(epoch, opt.n_epochs, train_loss))
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
        
        # save the model and history
        torch.save(net.state_dict(), "./model/{}_epoch_{}_{}".format(opt.model, opt.n_epochs, opt.optimizer))
        save_history(history, opt.model, "./model/{}_epoch_{}_{}".format(opt.model, opt.n_epochs, opt.optimizer))

    else:
        # Evaluate on clean and adversarial data
        print("eval model on test set")
        net.load_state_dict(torch.load("./model/{}_eps_{:.1f}".format(opt.model, opt.eps)))
        net.eval()
        report = {"n_test":0, "correct":0, "correct_fgm":0, "correct_pgd":0}

        if not os.path.exists("./data/attack_image/{}_eps_{:.1f}".format(opt.model, opt.eps)):
            os.makedirs("./data/attack_image/{}_eps_{:.1f}".format(opt.model, opt.eps))

        batch = 1
        for x, y, cap in test_data_loader:
            x, y = x.to(device),  y.to(device, dtype=torch.int64)
            x_fgm = fast_gradient_method(net, x, opt.eps, np.inf)
            x_pgd = projected_gradient_descent(net, x, opt.eps, 0.01, opt.pgd_nb_iter, np.inf)

            # only save 20 attacked images for efficiency
            if batch <= 10:
                label = cap[0]
                # save_image(x[0], 'data/attack_image/{}_eps_{:.1f}/{}_raw.png'.format(opt.model, opt.eps, label))
                save_image(x_pgd[0], './data/attack_image/{}_eps_{:.1f}/{}_pdg.png'.format(opt.model, opt.eps, label))
                save_image(x_fgm[0], './data/attack_image/{}_eps_{:.1f}/{}_fgsm.png'.format(opt.model, opt.eps, label))

            y_pred = net(x)  
            y_pred_fgm = net(x_fgm) 
            y_pred_pgd = net(x_pgd) 
            report["n_test"] += y.size(0)
            report["correct_fgm"] += y_pred_fgm.eq(y).sum().item()
            report["correct_pgd"] += y_pred_pgd.eq(y).sum().item()
            report["correct"] += y_pred.eq(y).sum().item()
            batch += 1

        print("test acc on clean examples (%): {:.3f}".format(report["correct"] / report["n_test"] * 100.0))
        print("test acc on FGM adversarial examples (%): {:.3f}".format(report["correct_fgm"] / report["n_test"] * 100.0))
        print("test acc on PGD adversarial examples (%): {:.3f}".format(report["correct_pgd"] / report["n_test"] * 100.0))


