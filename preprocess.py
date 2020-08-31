import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
from PIL import Image
from config import load_args
import numpy as np


class FFHG(Dataset):
    def __init__(self, args, mode='train', transforms=None):
        super(FFHG, self).__init__()
        self.mode = mode
        self.base_dir = 'E:/FFHG/dataset/{}'.format(mode)
        self.prepared_base_dir = 'E:/FFHG/prepared_dataset/{}'.format(mode)
        self.img_size = args.img_size
        # self.data_list = [os.path.join(self.base_dir, i) for i in os.listdir(self.base_dir)]
        self.data_list = os.listdir(self.base_dir)
        self.transforms = transforms

    def __getitem__(self, idx):
        filename = self.data_list[idx]

        data = {
            'img_filename': filename,
        }

        for size in self.img_size:
            size_dir = '{}_{}'.format(self.mode, size)
            img = Image.open(os.path.join(self.prepared_base_dir, size_dir, filename))

            if self.transforms is not None:
                img = self.transforms(img)

            data[str(size)] = img

        return data

    def __len__(self):
        return len(self.data_list)


def load_data(args):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    train_data = FFHG(args, mode='train', transforms=train_transforms)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_data = FFHG(args, mode='test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_loader, test_loader


def main():
    args = load_args()
    load_data(args)


# if __name__ == '__main__':
#     main()
