import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.augmentation import ToTensor


class VisDataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_path = os.listdir(os.path.join(data_dir, 'inference'))
        self.img_size = img_size
        self.labels = ['Street' , 'Resort', 'Feminine', 'Romantic', 'Modern', 'Classic', 'Sophisticated',
                       'Country', 'Genderless', 'Hippy', 'Sporty', 'Tomboy', 'Sexy', 'Manish', 'Retro',
                       'Oriental', 'Kidult', 'Military', 'Hiphop', 'Avantgarde', 'Preppy', 'Western', 'Punk']
        self.korean_names = ['스트리트', '리조트', '페미닌', '로맨틱', '모던', '클래식', '소피스트케이티드',
                             '컨트리', '젠더리스', '히피', '스포티', '톰보이', '섹시', '매니시', '레트로',
                             '오리엔탈', '키덜트', '밀리터리', '힙합', '아방가르드', '프레피', '웨스턴', '펑크']
        self.num_labels = len(self.labels)
        print('DATASET LOADED WITH %d INSTANCES' % len(self.img_path))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_full_path = os.path.join(self.data_dir, 'inference', self.img_path[idx])
        img = Image.open(img_full_path).convert('RGB')
        for k in range(self.num_labels):
            if self.korean_names[k] in self.img_path[idx]:
                label = k

        return {'img': img, 'img_size': self.img_size, 'label': label}

def collate_fn_vis(batch):
    fn = ToTensor(batch[0]['img_size'])
    images = [b['img'] for b in batch]
    images = torch.stack([fn(i) for i in images], 0)

    return images, [b['label'] for b in batch]
