import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image
import pandas as pd
import os

import numpy as np
from matplotlib import pyplot as plt


# Example training code:
# G = generator_oxford()
# D = discriminator_oxford()
#
# for l in [G, D]:         # if cuda enabled
#     l.cuda()
#
# train(G, D, label_flip=False, label_smooth=True)


# Oxford Pets constants
# Constants:
IMG_ROWS = 128
IMG_COLS = 128
CHANNELS = 3
NUM_IMGS = 7349
NUM_CLASSES = 20
LATENT_SIZE = 100
BATCH_SIZE = 10

# CIFAR10 constants
# Constants:
# IMG_ROWS = 32
# IMG_COLS = 32
# CHANNELS = 3
# NUM_IMGS = 60000
# NUM_CLASSES = 10
# LATENT_SIZE = 100
# BATCH_SIZE = 100



class OxfordPetsDataset(Dataset):
    """Oxford Pets Dataset"""

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(txt_file, sep=" ", header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img = self.annotations.iloc[idx, 0] + ".jpg"
        img_name = os.path.join(self.root_dir,
                                img)
        image = Image.open(img_name)
        image = image.convert('RGB')
        label = self.annotations.iloc[idx, 1:].as_matrix().astype('int')

        if self.transform:
            image = self.transform(image)


        return image, label


class generator_cifar(nn.Module):
    def __init__(self):
        super(generator_cifar, self).__init__()

        # Note: DCGAN paper architecture uses no dense layer in generator
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_SIZE, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # forward pass
    def forward(self, input):
        output = self.net(input)
        return output


class discriminator_cifar(nn.Module):
    def __init__(self):
        super(discriminator_cifar, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.net(input).view(-1, 1)
        return output



class generator_oxford(nn.Module):
    def __init__(self):
        super(generator_oxford, self).__init__()

        self.input_dim = LATENT_SIZE
        self.img_rows = IMG_ROWS
        self.img_cols = IMG_COLS

        # Note: DCGAN paper architecture uses no dense layer in generator
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # forward pass
    def forward(self, input):
        output = self.net(input)
        return output



class discriminator_oxford(nn.Module):
    def __init__(self):
        super(discriminator_oxford, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.net(input).view(-1, 1)
        return output









def train(G, D, epochs=300, label_flip=True, label_smooth=True, data="oxford"):
    criterion = nn.BCELoss().cuda()

    if data == "oxford":
        oxford = OxfordPetsDataset(txt_file="/home/ubuntu/data/oxford-pets/annotations/list.txt",
                                   root_dir="/home/ubuntu/data/oxford-pets/images/",
                                   transform=transforms.Compose([
                                    transforms.Resize(IMG_ROWS),
                                    transforms.CenterCrop(IMG_COLS),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        dataloader = DataLoader(oxford, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=4)

        dataset = dset.ImageFolder(root='/home/ubuntu/data/animal-faces/',
                                               transform=transforms.Compose([
                                                transforms.Resize(64),
                                                transforms.CenterCrop(64),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
    if data == "cats":
        dataset = dset.ImageFolder(root='/home/ubuntu/data/cats/',
                                       transform=transforms.Compose([
                                        transforms.Resize(IMG_ROWS),
                                        transforms.CenterCrop(IMG_COLS),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)

    real_x = torch.FloatTensor(BATCH_SIZE, CHANNELS, IMG_ROWS, IMG_COLS).cuda()
    label = torch.FloatTensor(BATCH_SIZE).cuda()
    noise = torch.FloatTensor(BATCH_SIZE, LATENT_SIZE).cuda()
    f_noise = torch.FloatTensor(100, LATENT_SIZE).cuda()

    real_x = Variable(real_x)
    label = Variable(label, requires_grad=False)
    noise = Variable(noise)
    f_noise = Variable(f_noise)

    optD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.99))
    optG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.99))

    # For saving imgs
    fixed_noise = torch.Tensor(100, LATENT_SIZE).normal_(0, 1)

    for epoch in range(epochs):
        for num_iters, batch_data in enumerate(dataloader, 0):

            # DISCRIMINATOR
            optD.zero_grad()
            x, _ = batch_data
            if x.size()[0] == BATCH_SIZE:
                real_x.data.resize_(x.size())
                real_x.data.copy_(x)
                pred_real = D(real_x)

                lab = np.ones(BATCH_SIZE).astype(int)
                if label_smooth:
                    lab = np.random.uniform(0.7, 1.2, BATCH_SIZE)
                if label_flip:
                    if (epoch % 3 == 0) and (epoch > 0):
                        lab = np.zeros(BATCH_SIZE).astype(int)

                label.data.copy_(torch.from_numpy(lab))
                d_loss_real = criterion(pred_real, label)
                d_loss_real.backward()

                noise.data.normal_(0, 1)
                noise = noise.view(-1, LATENT_SIZE, 1, 1)
                fake = G(noise)
                pred_fake = D(fake.detach())     # dont train generator

                lab = np.zeros(BATCH_SIZE).astype(int)
                if label_flip:
                    if (epoch % 3 == 0) and (epoch > 0):
                        lab = np.ones(BATCH_SIZE).astype(int)
                label.data.copy_(torch.from_numpy(lab))
                d_loss_fake = criterion(pred_fake, label)
                d_loss_fake.backward()

                d_loss_total = d_loss_real + d_loss_fake

                optD.step()


                # GENERATOR
                optG.zero_grad()
                pred_g = D(fake)
                label.data.fill_(1)

                g_loss = criterion(pred_g, label)

                g_loss.backward()
                optG.step()

            if num_iters % 100 == 0:
                print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch, num_iters, d_loss_total.data.cpu().numpy(),
                        g_loss.data.cpu().numpy())
                     )
        f_noise = f_noise.view(100, LATENT_SIZE, 1, 1)
        f_noise.data.copy_(fixed_noise)
        to_save = G(f_noise)

        # normalize=True important! Otherwise all images look dark
        vutils.save_image(to_save.data, '/home/ubuntu/photos/pytorch-dcgan-cats-2/%d.png' % (epoch), nrow=10,
                          normalize=True)
