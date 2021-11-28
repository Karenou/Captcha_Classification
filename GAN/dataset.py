import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class CaptchaDataset(Dataset):
    def __init__(self, base_path, target_size=(44,140)):
        
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(), 
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]) 
        ])
        
        self.images = glob.glob("%s/*" % base_path)
        self.labels = [c.split("/")[-1][:6] for c in self.images]

    def __getitem__(self, item):
        image = Image.open(self.images[item])  
        image = self.transform(image)   
        label = self.labels[item]
        return image, label

    def __len__(self):
        return len(self.images)