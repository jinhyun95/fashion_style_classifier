import os
import torch
from torch.utils.data import Dataset
from utils.augmentation import Augmentation, ToTensor
import numpy as np
from PIL import Image

# from tqdm import tqdm
# from scipy import io
# import random
# random.seed(1227)
# mat = io.loadmat('/data3/fashion/hipsterwars_Jan_2014.mat')
# print(len(mat['samples']))
# tr = 0
# va = 0
# te = 0
# split_idx = list(range(len(mat['samples'])))
# random.shuffle(split_idx)
# for idx in tqdm(range(len(mat['samples']))):
#     if split_idx[idx] < int(len(mat['samples']) * 0.7):
#         split = 'train'
#         tr += 1
#     elif split_idx[idx] < int(len(mat['samples']) * 0.9):
#         split = 'test'
#         te += 1
#     else:
#         split = 'val'
#         va += 1
#     split = os.path.join('/data3', 'fashion', 'hipsterwars', split)
#     np.save(os.path.join(split, mat['samples'][idx][0][1][0] + str(mat['samples'][idx][0][0][0])), mat['samples'][idx][0][6])
# print(tr, te, va)


class FashionDataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224), phase='train'):
        torch.random.manual_seed(928)
        self.phase = phase
        self.data_dir = os.path.join(data_dir, self.phase)
        self.img_path = os.listdir(self.data_dir)
        self.img_size = img_size
        self.labels = ['Bohemian', 'Goth', 'Hipster', 'Pinup', 'Preppy']
        self.one_hot_label_dict = {}
        for idx, label in enumerate(self.labels):
            self.one_hot_label_dict[label] = idx
        self.num_labels = len(self.labels)
        print('DATASET LOADED WITH %d INSTANCES' % len(self.img_path))
        self.inverse_dict = self.labels

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_full_path = os.path.join(self.data_dir, self.img_path[idx])
        img = Image.fromarray(np.load(img_full_path))
        label = self.img_path[idx].split('[')[0]
        label_idx = self.one_hot_label_dict[label]

        return {'img': img, 'label': label_idx, 'img_size': self.img_size}


def collate_fn_train(batch):
    fn = Augmentation(batch[0]['img_size'])
    images = [b['img'] for b in batch]
    images = torch.stack([fn(i) for i in images], 0)
    targets = torch.tensor([b['label'] for b in batch])

    return images, targets


def collate_fn_test(batch):
    fn = ToTensor(batch[0]['img_size'])
    images = [b['img'] for b in batch]
    images = torch.stack([fn(i) for i in images], 0)
    targets = torch.tensor([b['label'] for b in batch])

    return images, targets
