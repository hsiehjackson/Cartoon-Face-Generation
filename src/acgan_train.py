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
from dataset import CartoonDataset
from PIL import Image
from models_bce import ACGenerator, ACDiscriminator, ACGenerator_noconcat
from models_sn import ACDiscriminator_SN, ACDiscriminator_SNPJ


parser = argparse.ArgumentParser()
parser.add_argument("file_name",type=str)
parser.add_argument("--train_attr_file", type=str, default='./data/selected_cartoonset100k/cartoon_attr.txt')
parser.add_argument("--img_dir", type=str, default='./data/selected_cartoonset100k/images')
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--all_classes_dim",type=int, default=15)
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_layers", type=int, default=5)

parser.add_argument("--sample_interval", type=int, default=5, help="interval between image sampling")
parser.add_argument("--test_interval", type=int, default=10, help="interval between image testing")
parser.add_argument("--test_start", type=int, default=0, help="start test image")

parser.add_argument("--lambda_gp", type=int, default=1)
parser.add_argument("--lambda_aux", type=int, default=20)
parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value for disc. weights")
parser.add_argument("--loss_type", type=str,  default='WGGP', help='MM, NS, WGCP, WGGP WGDIV')
parser.add_argument("--generator_type",type=str, default='concat', help='concat, noconcat')
parser.add_argument("--discriminator_type",type=str, default='SN', help='noSN, SN, SNPJ')
parser.add_argument("--seed",type=int,default=422)

opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

os.makedirs("./saves/{}".format(opt.file_name),exist_ok=True)
os.makedirs("./saves/{}/train_sample".format(opt.file_name), exist_ok=True)
os.makedirs("./saves/{}/models".format(opt.file_name), exist_ok=True)

auxiliary_loss = torch.nn.BCELoss()

if opt.generator_type == 'concat':
    generator = ACGenerator(opt, n_filters = [512, 256, 128, 64, 32])
elif opt.generator_type == 'noconcat':
    generator = ACGenerator_noconcat(opt, n_filters = [512, 256, 128, 64, 32])

if opt.discriminator_type == 'noSN':
    discriminator = ACDiscriminator(opt, n_filters = [32, 64, 128, 256, 512])
elif opt.discriminator_type == 'SN':
    discriminator = ACDiscriminator_SN(opt, n_filters = [32, 64, 128, 256, 512])
elif opt.discriminator_type == 'SNPJ':
    discriminator = ACDiscriminator_SNPJ(opt, n_filters = [32, 64, 128, 256, 512])


if cuda:
    generator.cuda()
    discriminator.cuda()
    auxiliary_loss.cuda()

data_transform = transforms.Compose([
transforms.Resize(opt.img_size),
transforms.RandomHorizontalFlip(), 
transforms.ToTensor(),
transforms.Normalize([0.5]*opt.channels, [0.5]*opt.channels)])

# Configure data loader
train_dataset = CartoonDataset(attr_file=opt.train_attr_file, img_dir=opt.img_dir, transform=data_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,shuffle=True,num_workers=16,drop_last=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay = 1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay = 1e-4)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

np.random.seed(opt.seed)
fixed_z = FloatTensor(np.random.normal(0, 1, (12 ** 2, opt.latent_dim)))
    
def sample_image(n_row, epoch):
    hair_labels = np.eye(6)[np.array([num  for num in [0,0,1,1,2,2,3,3,4,4,5,5] for _ in range(n_row)])]
    eye_labels = np.eye(4)[np.array([num for _ in range(int(n_row/2)) for num in ([0]*6+[1]*6+[2]*6+[3]*6)])]
    face_labels = np.eye(3)[np.array([num for _ in range(n_row) for num in [0,0,1,1,2,2] * 2])]
    glasses_labels = np.eye(2)[np.array([num for _ in range(n_row) for num in [0,1] * 6])]
    condition = FloatTensor(np.concatenate((hair_labels,eye_labels,face_labels,glasses_labels),axis=1))
    gen_imgs = generator(fixed_z, condition)
    save_image(gen_imgs.data, "./saves/{}/train_sample/ep-{}_seed-{}.png".format(opt.file_name,epoch,opt.seed), nrow=n_row, normalize=True)


def compute_gradient_penalty_GP(D, real_samples, fake_samples, condition):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).cuda()
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    d_interpolates, _ = D(interpolates, condition)
    fake = torch.ones(d_interpolates.shape).cuda()
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty *  opt.lambda_gp


def compute_gradient_penalty_DIV(real_pred, fake_pred, real_imgs, fake_imgs):
    p = 6
    real_grad_out = Variable(FloatTensor(real_imgs.size(0), 1).fill_(1.0), requires_grad=False)
    real_grad = torch.autograd.grad(real_pred, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    fake_grad_out = Variable(FloatTensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
    fake_grad = torch.autograd.grad(fake_pred, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True)[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    div_gp = torch.mean(real_grad_norm + fake_grad_norm) * opt.lambda_gp
    return div_gp
# ----------
#  Training
# ----------

plot = {
    'd_real_loss':[],
    'd_fake_loss':[],
    'g_adv_loss':[],
    'g_aux_loss':[],
    'd_acc':[],
}


for epoch in range(opt.n_epochs):

    batch_num = 0
    plot['d_real_loss'].append(0.0)
    plot['d_fake_loss'].append(0.0)
    plot['g_aux_loss'].append(0.0)
    plot['g_adv_loss'].append(0.0)
    plot['d_acc'].append(0.0)


    for i, sample in enumerate(train_dataloader):

        batch_size = sample['image'].shape[0]

        # Configure input
        real_imgs = Variable(sample['image'].type(FloatTensor), requires_grad=True)
        true_vec = sample['true_vec'].type(FloatTensor)
        # -----------------
        #  Train Generator
        # -----------------
        
        z = torch.randn(batch_size, opt.latent_dim).cuda()
        gen_imgs = generator(z, true_vec)
        # Loss measures generator's ability to fool the discriminator            
        validity, pred_labels = discriminator(gen_imgs, true_vec)
        #g_adv_loss = adversarial_loss(validity, valid)
        if opt.loss_type == 'MM':
            g_adv_loss = torch.mean(torch.log(1 - torch.sigmoid(validity) + 1e-5))
            g_aux_loss = auxiliary_loss(pred_labels, true_vec)
        elif opt.loss_type == 'NS':
            g_adv_loss  = - torch.mean(torch.log(torch.sigmoid(validity) + 1e-5))
            g_aux_loss = auxiliary_loss(pred_labels, true_vec)    
        elif opt.loss_type == 'WGCP' or opt.loss_type == 'WGGP' or opt.loss_type == 'WGDIV':
            g_adv_loss = - torch.mean(validity)
            g_aux_loss = auxiliary_loss(pred_labels, true_vec)  * opt.lambda_aux
        
        g_loss = (g_adv_loss + g_aux_loss)
        optimizer_G.zero_grad()
        g_loss.backward(retain_graph = True)
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        z = torch.randn(batch_size, opt.latent_dim).cuda()
        gen_imgs = generator(z, true_vec)
        
        real_pred, real_labels = discriminator(real_imgs, true_vec)
        if opt.loss_type == 'WGDIV':
            fake_pred, fake_labels = discriminator(gen_imgs, true_vec)
        else:
            fake_pred, fake_labels = discriminator(gen_imgs.detach(), true_vec)


        if opt.loss_type == 'MM':
            d_real_adv_loss = torch.mean(torch.log(1 - torch.sigmoid(real_pred) + 1e-5))
            d_real_aux_loss = auxiliary_loss(real_labels, true_vec)    
            d_fake_adv_loss = torch.mean(torch.log(1 - torch.sigmoid(fake_pred) + 1e-5))
            d_fake_aux_loss = auxiliary_loss(fake_labels, true_vec)
        elif opt.loss_type == 'NS':
            d_real_adv_loss  = - torch.mean(torch.log(torch.sigmoid(real_pred) + 1e-5))
            d_real_aux_loss = auxiliary_loss(real_labels, true_vec)    
            d_fake_adv_loss  = - torch.mean(torch.log(torch.sigmoid(fake_pred) + 1e-5))
            d_fake_aux_loss = auxiliary_loss(fake_labels, true_vec)   
        elif opt.loss_type == 'WGCP' or opt.loss_type == 'WGGP' or opt.loss_type=='WGDIV':
            d_real_adv_loss = - torch.mean(real_pred)
            d_real_aux_loss = auxiliary_loss(real_labels, true_vec) * opt.lambda_aux
            d_fake_adv_loss = torch.mean(fake_pred)
            d_fake_aux_loss = auxiliary_loss(fake_labels, true_vec) * opt.lambda_aux
        
        if opt.loss_type == 'WGGP':
            gradient_penalty = compute_gradient_penalty_GP(discriminator, real_imgs, gen_imgs, true_vec)

        if opt.loss_type == 'WGDIV':
            gradient_penalty = compute_gradient_penalty_DIV(real_pred, fake_pred, real_imgs, gen_imgs)

        # Total discriminator loss
        d_real_loss = d_real_adv_loss + d_real_aux_loss 
        d_fake_loss = d_fake_adv_loss + d_fake_aux_loss
        optimizer_D.zero_grad()
        d_real_loss.backward(retain_graph=True)
        d_fake_loss.backward(retain_graph=True)

        if opt.loss_type == 'WGGP' or opt.loss_type=='WGDIV':
            gradient_penalty.backward(retain_graph=True)
        
        optimizer_D.step()

        if opt.loss_type == 'WGCP':
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)


        # Calculate discriminator accuracy
        pred = np.concatenate([real_labels.data.cpu().numpy(), fake_labels.data.cpu().numpy()], axis=0).flatten()
        gt = np.concatenate([true_vec.data.cpu().numpy(), true_vec.data.cpu().numpy()], axis=0).flatten()
        pred[pred>=0.5] = 1.0
        pred[pred<0.5] = 0.0
        d_acc = np.mean(pred == gt)

        plot['d_real_loss'][epoch] += d_real_loss.item()
        plot['d_fake_loss'][epoch] += d_fake_loss.item()
        plot['g_adv_loss'][epoch] += g_adv_loss.item()
        plot['g_aux_loss'][epoch] += g_aux_loss.item()
        plot['d_acc'][epoch] += d_acc

        print("[Epoch %d/%d] [Batch %d/%d] [D real: %f, fake: %f, acc: %d%%] [G adv: %f, aux: %f]"
              % (epoch+1, opt.n_epochs, i, len(train_dataloader), plot['d_real_loss'][-1]/(i+1), plot['d_fake_loss'][-1]/(i+1),
                100 * plot['d_acc'][-1]/(i+1), plot['g_adv_loss'][-1]/(i+1), plot['g_aux_loss'][-1]/(i+1)), end='\r')

        batch_num += 1

    for k, v in plot.items():
        plot[k][epoch] = plot[k][epoch] / batch_num

    print("[Epoch %d/%d] [D real: %f, fake: %f, acc: %d%%] [G adv: %f, aux: %f]"
        % (epoch+1, opt.n_epochs, plot['d_real_loss'][-1], plot['d_fake_loss'][-1],
            100 * plot['d_acc'][-1],  plot['g_adv_loss'][-1], plot['g_aux_loss'][-1]))

    if ((epoch+1) >= opt.test_start) and ((epoch+1) % opt.sample_interval == 0):
        sample_image(n_row=12, epoch=epoch+1)

    with open('./saves/{}/plot.json'.format(opt.file_name), 'w') as f:
        json.dump(plot,f)


    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict()
            }, './saves/{}/models/epoch-{}.pth'.format(opt.file_name,epoch+1))
