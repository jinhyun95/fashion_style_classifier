import os
import json
from PIL import Image
import time
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import Dataset
from utils.augmentation import Augmentation, ToTensor


class FashionDataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224), phase='train'):
        self.phase = phase
        self.data_dir = data_dir
        self.fashion_labels = dict({
            ('스트리트', 0),
            ('리조트', 1),
            ('페미닌', 2),
            ('로맨틱', 3),
            ('모던', 4),
            ('클래식', 5),
            ('소피스트케이티드', 6),
            ('컨트리', 7),
            ('젠더리스', 8),
            ('히피', 9),
            ('스포티', 10),
            ('톰보이', 11),
            ('섹시', 12),
            ('매니시', 13),
            ('레트로', 14),
            ('오리엔탈', 15),
            ('키덜트', 16),
            ('밀리터리', 17),
            ('힙합', 18),
            ('아방가르드', 19),
            ('프레피', 20),
            ('웨스턴', 21),
            ('펑크', 22),
        })
        self.inverse_dict = ['Street' , 'Resort', 'Feminine', 'Romantic', 'Modern', 'Classic', 'Sophisticated',
                             'Country', 'Genderless', 'Hippy', 'Sporty', 'Tomboy', 'Sexy', 'Manish', 'Retro',
                             'Oriental', 'Kidult', 'Military', 'Hiphop', 'Avantgarde', 'Preppy', 'Western', 'Punk']
        image_names = os.path.join(data_dir, self.phase + '.txt')
        with open(image_names, 'r') as image_names:
            self.images = [n.strip() for n in image_names.readlines()]
        loading = True
        while loading:
            try:
                with open(os.path.join(data_dir, 'image_names.pickle'), 'rb') as handle:
                    self.image_filenames = pickle.load(handle)
                loading = False
            except:
                time.sleep(1)

        self.img_size = img_size
        self.num_labels = len(self.fashion_labels)
        print('DATASET LOADED WITH %d INSTANCES' % len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, self.images[idx], self.image_filenames[self.images[idx]])).convert('RGB')
        jsonpath = os.path.join(self.data_dir, self.images[idx], self.images[idx] + '.json')
        with open(jsonpath) as f:
            styles = json.load(f)['데이터셋 정보']['데이터셋 상세설명']['라벨링']['스타일'][0]
        main_style = styles['스타일']
        if main_style.startswith('키치'):
            main_style = '키덜트'
        main_style = self.fashion_labels[main_style]
        label = [0. for _ in self.fashion_labels]
        label[main_style] = 1.
        if '서브스타일' in styles.keys():
            sub_style = styles['서브스타일']
            if sub_style.startswith('키치'):
                sub_style = '키덜트'
            sub_style = self.fashion_labels[sub_style]
            label[sub_style] = 1.

        return {'img': img, 'label': torch.tensor(label, dtype=torch.float32), 'img_size': self.img_size,
                'main_style': main_style}


def collate_fn_train(batch):
    fn = Augmentation(batch[0]['img_size'])
    images = [b['img'] for b in batch]
    images = torch.stack([fn(i) for i in images], 0)
    targets = torch.stack([b['label'] for b in batch], 0)
    main_style = torch.tensor([b['main_style'] for b in batch])

    return images, targets, main_style


def collate_fn_test(batch):
    fn = ToTensor(batch[0]['img_size'])
    images = [b['img'] for b in batch]
    images = torch.stack([fn(i) for i in images], 0)
    targets = torch.stack([b['label'] for b in batch], 0)
    main_style = torch.tensor([b['main_style'] for b in batch])

    return images, targets, main_style


# dir = '/data5/fashion/K-fashion'
# image_names = dict()
# for i in tqdm(os.listdir(dir)):
#     if os.path.isdir(os.path.join(dir, i)):
#         for jpg in os.listdir(os.path.join(dir, i)):
#             if jpg.lower().endswith('.jpg'):
#                 image_names[i] = jpg
# with open(os.path.join(dir, 'image_names.pickle'), 'wb') as handle:
#     pickle.dump(image_names, handle)

