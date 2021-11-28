import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd


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

def save_history(history, model_name, save_path):
    df = pd.DataFrame.from_dict(history)
    df.to_csv(save_path + "_loss.csv", header=True)
    
    plt.figure(figsize=(6,4))
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Loss")
    plt.title("Adversarial Attack Trained on %s" % model_name)
    plt.savefig(save_path + "_loss.png")
    plt.close()

    