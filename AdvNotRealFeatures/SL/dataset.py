import os
import torch
from torchvision import transforms

class Crafted_CIFAR10_Training_Set(torch.utils.data.Dataset):
    def __init__(self, transform, version):
        self.transform = transform    
        self.to_img = transforms.ToPILImage()
        if version == 'robust-cifar10':
            data_path = '../data/release_datasets/d_robust_CIFAR'
        elif version == 'non-robust-cifar10':
            data_path = '../data/release_datasets/d_non_robust_CIFAR'
        self.data = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_ims")))
        self.targets = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_lab")))

        
    def __getitem__(self, index):
        if self.transform:
            img = self.to_img(self.data[index])
            img = self.transform(img)
        else:
            img = self.data[index]
        
        return img, self.targets[index]

    def __len__(self):
        return len(self.targets)