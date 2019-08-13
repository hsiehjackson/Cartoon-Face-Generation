import argparse
import os
import numpy as np
import math
import json

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models_bce import ACGenerator, ACGenerator_noconcat
from dataset import CartoonDataset
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--folder",type=str,default='best')
parser.add_argument("--model",type=str,default='model/model.pth')
parser.add_argument("--test_attr_file", type=str, default='./test/sample_fid_testing_labels.txt')
parser.add_argument("--img_dir", type=str, default='./data/selected_cartoonset100k/images')
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_layers", type=int, default=5)
parser.add_argument("--all_classes_dim",type=int, default=15)
parser.add_argument("--seed",type=int,default=422)

opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False


os.makedirs("./saves/{}/test_images/".format(opt.folder), exist_ok=True)
os.makedirs("./saves/{}/test_sample/".format(opt.folder), exist_ok=True)

def norm_ip(img, min, max):
    img = img.clamp_(min=min, max=max)
    img = img.add_(-min).div_(max - min + 1e-5)
    return img

def norm_range(t):
    return norm_ip(t, float(t.min()), float(t.max()))

generator = ACGenerator(opt, n_filters = [1024, 512, 256, 128, 64])
generator.load_state_dict(torch.load(opt.model)['generator'])

if cuda:
    generator.cuda()

data_transform = transforms.Compose([
transforms.Resize(opt.img_size),
transforms.ToTensor(),
transforms.Normalize([0.5]*opt.channels, [0.5]*opt.channels)])

test_dataset = CartoonDataset(attr_file=opt.test_attr_file, img_dir=opt.img_dir, transform=False, test=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,shuffle=False)
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

torch.manual_seed(opt.seed)

for i, sample in enumerate(tqdm(test_dataloader)):
    batch_size = len(sample['img_id'])
    z = torch.randn(batch_size, opt.latent_dim).cuda()
    true_vec = sample['true_vec'].type(FloatTensor)
    gen_imgs = generator(z, true_vec)
    for i_d, gen_img in zip(sample['img_id'], gen_imgs):
        t_img = norm_range(gen_img.data)
        n_img = t_img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(n_img)
        im.save("./saves/{}/test_images/{}".format(opt.folder,i_d))

# sample image
n_row = 12
np.random.seed(opt.seed)
fixed_z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))

hair_labels = np.eye(6)[np.array([num  for num in [0,0,1,1,2,2,3,3,4,4,5,5] for _ in range(n_row)])]
eye_labels = np.eye(4)[np.array([num for _ in range(int(n_row/2)) for num in ([0]*6+[1]*6+[2]*6+[3]*6)])]
face_labels = np.eye(3)[np.array([num for _ in range(n_row) for num in [0,0,1,1,2,2] * 2])]
glasses_labels = np.eye(2)[np.array([num for _ in range(n_row) for num in [0,1] * 6])]
condition = FloatTensor(np.concatenate((hair_labels,eye_labels,face_labels,glasses_labels),axis=1))
gen_imgs = generator(fixed_z, condition)
save_image(gen_imgs.data, "./saves/{}/test_sample/best_seed-{}.png".format(opt.folder,opt.seed), nrow=n_row, normalize=True)
