import os
import unicodedata
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.augmentation import Augmentation, ToTensor


class FashionDataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224), phase='train', fssplit=None):
        self.phase = phase
        if fssplit is not None and phase != 'test':
            dataset_split_path = os.path.join(data_dir, 'imbalanced_splits', 'split_%d_%s.csv' % (fssplit, phase))
        else:
            dataset_split_path = os.path.join(data_dir, self.phase + '.csv')
        self.data_dir = data_dir
        self.img_path = []
        with open(dataset_split_path, 'r') as img_path_file:
            names = img_path_file.readlines()
            for fn in names:
                if os.path.exists(os.path.join(self.data_dir, unicodedata.normalize('NFD', fn.strip()))):
                    self.img_path.append(unicodedata.normalize('NFD', fn.strip()))
                else:
                    print('%s DOES NOT EXIST IN DIR' % fn.strip())
        self.img_size = img_size
        self.labels = os.listdir(os.path.join(self.data_dir, 'dataset'))
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
        img = Image.open(img_full_path).convert('RGB')
        label = self.img_path[idx].split("/")[1]
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
