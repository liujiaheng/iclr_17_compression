from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from data_loader.datasets import Dataset
import torch
import pdb


class Datasets(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def get_loader(train_data_dir, test_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    test_dataset = Datasets(test_data_dir, image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def get_train_loader(train_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    torch.manual_seed(3334)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True)
    return train_dataset, train_loader

class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)

def build_dataset():
    train_set_dir = '/data1/liujiaheng/data/compression/Flick_patch/'
    dataset, dataloader = get_train_loader(train_set_dir, 256, 4)
    for batch_idx, (image, path) in enumerate(dataloader):
        pdb.set_trace()


if __name__ == '__main__':
    build_dataset()
