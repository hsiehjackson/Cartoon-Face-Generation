import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class CartoonDataset(Dataset):
    def __init__(self, attr_file, img_dir, transform=None, test=False):
        self.annotations = pd.read_csv(attr_file,skiprows=1)
        self.img_dir = img_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.test:
            img_name = "{}.png".format(idx)
            img_vec = str(self.annotations.iloc[idx,0]).split()
        else:
            img_name = str(self.annotations.iloc[idx,0]).split()[0]
            image = Image.open(os.path.join(self.img_dir, img_name))
            img_vec = str(self.annotations.iloc[idx,0]).split()[1:]


        img_vec = [int(i) for i in img_vec]
        true_labels = {}
        true_labels['hair'] =int(img_vec[:6].index(1))
        true_labels['eye'] = int(img_vec[6:10].index(1))
        true_labels['face'] = int(img_vec[10:13].index(1))
        true_labels['glasses'] = int(img_vec[13:15].index(1))

        gen_img_vec = np.eye(6)[np.random.choice(6)].tolist() + \
                       np.eye(4)[np.random.choice(4)].tolist() + \
                       np.eye(3)[np.random.choice(3)].tolist() + \
                       np.eye(2)[np.random.choice(2)].tolist()

        gen_img_vec = [int(i) for i in gen_img_vec]

        gen_labels = {}
        gen_labels['hair'] =int(gen_img_vec[:6].index(1))
        gen_labels['eye'] = int(gen_img_vec[6:10].index(1))
        gen_labels['face'] = int(gen_img_vec[10:13].index(1))
        gen_labels['glasses'] = int(gen_img_vec[13:15].index(1))


        if self.test:
            sample = {'img_id': img_name, 'true_labels': true_labels, 'true_vec':torch.LongTensor(img_vec),}
        else:
            sample = {'img_id': img_name, 'image': image, 
                      'true_labels': true_labels, 'true_vec':torch.FloatTensor(img_vec), 
                      'gen_labels': gen_labels, 'gen_vec':torch.FloatTensor(gen_img_vec)}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
