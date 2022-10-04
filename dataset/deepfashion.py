import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.augmentation import Augmentation, ToTensor


class FashionDataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224), phase='train'):
        self.phase = phase
        self.data_dir = data_dir
        image_names = os.path.join(data_dir, 'Attr_Predict', 'Anno_fine', self.phase + '.txt')
        with open(image_names, 'r') as image_names:
            self.images = [n.strip() for n in image_names.readlines()]
        annotations = os.path.join(data_dir, 'Attr_Predict', 'Anno_fine', self.phase + '_attr.txt')
        with open(annotations, 'r') as annotations:
            self.annotations = [[float(x) for x in a.strip().split(' ')] for a in annotations.readlines()]
        self.img_size = img_size
        self.num_labels = len(self.annotations[0])
        print('DATASET LOADED WITH %d INSTANCES' % len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_full_path = os.path.join(self.data_dir, self.images[idx])
        img = Image.open(img_full_path).convert('RGB')
        label = self.annotations[idx]

        return {'img': img, 'label': label, 'img_size': self.img_size}


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
