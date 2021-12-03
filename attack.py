import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from cleverhans.torch.attacks.noise import noise
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)

from model import get_optimizer, get_model, simba_single
from dataset import CAPTCHA
from util import make_dir


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="res50", help="Pre-trained model.")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for model during training.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the optimizer during training")
parser.add_argument("--eps", type=float, default=0.2, help="Total epsilon for FGM and PGD attacks.")
parser.add_argument("--sim_eps", type=float, default=0.2, help="Total epsilon for noise and simba_single attacks.")
parser.add_argument("--sparse_eps", type=float, default=20, help="Total epsilon for sparse l1 descent attacks.")
parser.add_argument("--pgd_nb_iter", type=int, default=50, help="Number of iteration for PGD update.")
opt = parser.parse_args()
print(opt)


if __name__ == "__main__":
    # Load test data
    transform = transforms.Compose([transforms.ToTensor()])  
    test_data = CAPTCHA('./data/test/', mode="attack", transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False, drop_last=True)

    # Instantiate model
    net = get_model(opt.model)

    print("GPU available: ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()

    optimizer = get_optimizer(opt, net)

    print("generate adversarial examples on test set")
    checkpoint = torch.load("./model/{}.pth".format(opt.model))
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    base_save_path = "./data/attack_image/{}".format(opt.model)
    make_dir(base_save_path)
    make_dir("{}/noise_eps_{:.1f}".format(base_save_path, opt.sim_eps))
    make_dir("{}/simba_eps_{:.1f}".format(base_save_path, opt.sim_eps))
    make_dir("{}/fgsm_eps_{:.1f}".format(base_save_path, opt.eps))
    make_dir("{}/pgd_eps_{:.1f}".format(base_save_path, opt.eps))
    make_dir("{}/sparse_eps_{:.1f}".format(base_save_path, opt.sparse_eps))

    batch = 1
    for x, y, cap in test_data_loader:
        x, y = x.to(device),  y.to(device, dtype=torch.int64)
        x_noise = noise(x, eps=opt.sim_eps, order=np.inf)
        x_sim = simba_single(net, x, y, num_iters=10000, epsilon=opt.sim_eps)
        x_fgm = fast_gradient_method(net, x, opt.eps, np.inf)
        x_pgd = projected_gradient_descent(net, x, opt.eps, 0.01, opt.pgd_nb_iter, np.inf)
        x_sparse = sparse_l1_descent(net, x, eps=opt.sparse_eps, nb_iter=opt.pgd_nb_iter)

        # if batch % 10 == 0:
        print("batch %d" % batch)
        
        label = cap[0]
        save_image(x_noise[0], "{}/noise_eps_{:.1f}/{}.png".format(base_save_path, opt.sim_eps, label))
        save_image(x_sim[0], "{}/simba_eps_{:.1f}/{}.png".format(base_save_path, opt.sim_eps, label))
        save_image(x_fgm[0], "{}/fgsm_eps_{:.1f}/{}.png".format(base_save_path, opt.eps, label))
        save_image(x_pgd[0], "{}/pgd_eps_{:.1f}/{}.png".format(base_save_path, opt.eps, label))
        save_image(x_sparse[0], "{}/sparse_eps_{:.1f}/{}.png".format(base_save_path, opt.sparse_eps, label))

        batch += 1

