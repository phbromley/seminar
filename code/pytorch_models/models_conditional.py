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
import sys
import psutil
import gc
import time

import numpy as np
from matplotlib import pyplot as plt

from networks_conditional import *


# Example code for training:
# infogan_mnist = InfoGAN(channels=1, img_rows=28, img_cols=28, bs=128,
#                         nz=62, cat=10, cont=2, cont_lambda=0.1, data="mnist", data_path="/home/ubuntu/data/")
# cdcgan_cifar = ConditionalDCGAN(channels=3, img_rows=32, img_cols=32, bs=100,
#                                 nz=100, num_classes=10, data="cifar")
# infogan_mnist.train(epochs=50, save_model=True, plot_loss=True)
# cdcgan_cifar.train(epochs=100, save_model=True, plot_loss=True)


# Much inspiration taken from: https://github.com/pianomania/infoGAN-pytorch/blob/master/trainer.py
class InfoGAN():
    def __init__(self, channels, img_rows, img_cols, bs, nz, cat, cont, cont_lambda, data="mnist"):

        self.nz = nz
        self.cat = cat
        self.cont = cont
        self.cont_lambda = cont_lambda
        self.bs = bs
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.latent_size = nz + cat + cont
        self.data = data

        if (self.data == "mnist") or (self.data == "fashion"):
            self.G = infogan_generator_mnist(self.latent_size, self.channels)
            self.D_Q_body = infogan_d_q_body_mnist(self.channels)
            self.D = infogan_discriminator()
            self.Q = infogan_q(self.cat, self.cont)

            if self.data == "mnist":
                dataset = dset.MNIST(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_rows),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
            else:
                dataset = dset.FashionMNIST(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_rows),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

        if self.data == "cifar":
            self.G = infogan_generator_cifar(self.latent_size, self.channels)
            self.D_Q_body = infogan_d_q_body_cifar(self.channels)
            self.D = infogan_discriminator()
            self.Q = infogan_q(self.cat, self.cont)

            dataset = dset.CIFAR10(root=data_path, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(img_rows),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


        if self.data == "cats":
            self.G = infogan_generator_cats(self.latent_size, self.channels)
            self.D_Q_body = infogan_d_q_body_cats(self.channels)
            self.D = infogan_discriminator()
            self.Q = infogan_q(self.cat, self.cont)

            dataset = dset.ImageFolder(root='/home/ubuntu/data/cats/',
                                           transform=transforms.Compose([
                                            transforms.Resize(self.img_rows),
                                            transforms.CenterCrop(self.img_cols),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))


        self.dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        self.criterionDG = nn.BCELoss().cuda()
        self.criterionQcat = nn.CrossEntropyLoss().cuda()
        self.criterionQcont = log_gaussian()

        self.optD = optim.Adam([{'params':self.D_Q_body.parameters()},
                                {'params':self.D.parameters()}], lr=2e-4, betas=(0.5, 0.99))
        self.optG = optim.Adam([{'params':self.G.parameters()},
                                {'params':self.Q.parameters()}], lr=1e-3, betas=(0.5, 0.99))

        for l in [self.G, self.D_Q_body, self.D, self.Q]:
            l.cuda()
            l.apply(self.weights_init)

        self.train_hist = {}
        self.train_hist['d_loss'] = []
        self.train_hist['g_loss'] = []
        self.train_hist['q_loss'] = []
        self.train_hist['epoch_time'] = []
        self.train_hist['total_time'] = []


    def to_categorical(self, y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]


    def create_fixed_inputs(self):
        # For saving imgs
        np_fixed_noise = np.random.normal(0, 1, (self.cat, self.nz))
        fixed_noise = torch.from_numpy(np.repeat(np_fixed_noise, 9, axis=0)).float()

        np_fixed_cat = self.to_categorical(np.arange(0, self.cat, 1), self.cat)
        fixed_cat = torch.from_numpy(np.repeat(np_fixed_cat, 9, axis=0)).float()

        np_cont1 = np.concatenate([np.arange(-2.0, 2.5, 0.5).reshape(9, 1), np.zeros((9, 1))], axis=1)
        fixed_cont1 = torch.from_numpy(np.tile(np_cont1, (self.cat, 1))).float()

        np_cont2 = np.concatenate([np.zeros((9, 1)), np.arange(-2.0, 2.5, 0.5).reshape(9, 1)], axis=1)
        fixed_cont2 = torch.from_numpy(np.tile(np_cont2, (self.cat, 1))).float()

        fixed_noise = Variable(fixed_noise.cuda())
        fixed_cat = Variable(fixed_cat.cuda())
        fixed_cont1 = Variable(fixed_cont1.cuda())
        fixed_cont2 = Variable(fixed_cont2.cuda())

        return fixed_noise, fixed_cat, fixed_cont1, fixed_cont2


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    def initialize_variables(self):
        real_x = torch.FloatTensor(self.bs, self.channels, self.img_rows, self.img_cols).cuda()
        label = torch.FloatTensor(self.bs).cuda()
        noise = torch.FloatTensor(self.bs, self.nz).cuda()
        cat_c = torch.FloatTensor(self.bs, self.cat).cuda()
        cont_c = torch.FloatTensor(self.bs, self.cont).cuda()
        self.real_x = Variable(real_x)
        self.label = Variable(label)
        self.noise = Variable(noise)
        self.cat_c = Variable(cat_c)
        self.cont_c = Variable(cont_c)


    def batch_resize(self, x):
        batch = x.size(0)
        self.real_x.data.resize_(x.size())
        self.label.data.resize_(batch)
        self.noise.data.resize_(batch, self.nz)
        self.cat_c.data.resize_(batch, self.cat)
        self.cont_c.data.resize_(batch, self.cont)



    def calc_d_loss(self, real_or_fake):
        if real_or_fake == "real":
            self.label.data.fill_(1)
            body_real = self.D_Q_body(self.real_x)
            pred = self.D(body_real)
        else:
            self.label.data.fill_(0)
            body_fake = self.D_Q_body(self.fake_img.detach())
            pred = self.D(body_fake)

        return self.criterionDG(pred, self.label)



    def calc_g_q_loss(self, cat_labs):
        # G loss (real or fake)
        self.label.data.fill_(1)
        body_g_q = self.D_Q_body(self.fake_img)
        g_out = self.D(body_g_q)
        g_loss = self.criterionDG(g_out, self.label)

        # Q loss (latent codes)
        q_cat, q_mean, q_variance = self.Q(body_g_q)
        # convert from one hot to single int for loss calculation
        cat_from_onehot = np.where(cat_labs == 1)[1]
        cat_label = Variable(torch.LongTensor(cat_from_onehot).cuda())
        cat_loss = self.criterionQcat(q_cat, cat_label)
        cont_loss = self.criterionQcont(q_mean, q_variance, self.cont_c)*self.cont_lambda

        return g_loss, cat_loss, cont_loss

    def plot_loss(self, path):
        x = range(len(self.train_hist['d_loss']))
        d_loss_hist = self.train_hist['d_loss']
        g_loss_hist = self.train_hist['g_loss']
        q_loss_hist = self.train_hist['q_loss']
        plt.plot(x, d_loss_hist, label='d_loss')
        plt.plot(x, g_loss_hist, label='g_loss')
        plt.plot(x, q_loss_hist, label='q_loss')
        plt.legend(loc=4)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.savefig(path)
        plt.close()

    def save_imgs(self, f_noise, f_cat, f_cont1, f_cont2, epoch):
        to_save1 = self.G(f_noise, f_cat, f_cont1)
        to_save2 = self.G(f_noise, f_cat, f_cont2)

        # normalize=True important! Otherwise all images look dark
        img_path1 = '/home/ubuntu/photos/pytorch-infogan-'+str(self.data)+'/%d_%d.png' % (epoch, 1)
        img_path2 = '/home/ubuntu/photos/pytorch-infogan-'+str(self.data)+'/%d_%d.png' % (epoch, 2)
        vutils.save_image(to_save1.data, img_path1, nrow=9, normalize=True)
        vutils.save_image(to_save2.data, img_path2, nrow=9, normalize=True)


    def update_train_hist(self, d_loss, g_loss, q_loss):
        self.train_hist['d_loss'].append(d_loss.data[0])
        self.train_hist['g_loss'].append(g_loss.data[0])
        self.train_hist['q_loss'].append(q_loss.data[0])


    def train(self, epochs=100, save_model=False, plot_loss=False):

        self.initialize_variables()

        fixed_noise, fixed_cat, fixed_cont1, fixed_cont2 = self.create_fixed_inputs()

        start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            for num_iters, batch_data in enumerate(self.dataloader, 0):

                x, _ = batch_data

                self.batch_resize(x)

                # Get real image batch, noise batch, latent variable batch, and fake image batch
                self.real_x.data.copy_(x)
                self.noise.data.normal_(0, 1)
                cat_labs = self.to_categorical(np.random.randint(0, self.cat, x.size(0)), self.cat)
                self.cat_c.data.copy_(torch.from_numpy(cat_labs))
                self.cont_c.data.uniform_(-1, 1)
                self.fake_img = self.G(self.noise, self.cat_c, self.cont_c)


                # Discriminator
                self.optD.zero_grad()
                d_loss_real = self.calc_d_loss("real")
                d_loss_real.backward()
                d_loss_fake = self.calc_d_loss("fake")
                d_loss_fake.backward()

                d_loss_total = d_loss_real + d_loss_fake

                self.optD.step()


                # Generator and Q
                self.optG.zero_grad()
                g_loss, cat_loss, cont_loss = self.calc_g_q_loss(cat_labs)

                g_loss_total = g_loss + cat_loss + cont_loss
                g_loss_total.backward()

                self.optG.step()

                if plot_loss:
                    self.update_train_hist(d_loss_total, g_loss, cat_loss + cont_loss)

                if num_iters % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                            epoch, num_iters, d_loss_total.data.cpu().numpy(),
                            g_loss_total.data.cpu().numpy())
                         )

            self.train_hist['epoch_time'].append(time.time() - epoch_start_time)
            self.save_imgs(fixed_noise, fixed_cat, fixed_cont1, fixed_cont2, epoch)

        print("Training is complete!")
        if save_model:
            path = "/home/ubuntu/saved_models/pytorch-infogan-" + str(self.data)
            print("Saving Model...")
            torch.save(self.G.state_dict(), path + "g.pth")
            torch.save(self.D.state_dict(), path + "d.pth")
            torch.save(self.D_Q_body.state_dict(), path + "dq.pth")
            torch.save(self.Q.state_dict(), path + "q.pth")


        self.train_hist['total_time'].append(time.time() - start_time)
        print("Total training time (%d epochs): " % epochs + str(self.train_hist['total_time'][0]))
        avg_epoch_time = np.mean(self.train_hist['epoch_time'])
        print("Average time for each epoch: %.2f" % avg_epoch_time)

        if plot_loss:
            self.plot_loss("/home/ubuntu/loss-plots/pytorch-infogan-" + str(self.data) + ".png")







class ConditionalDCGAN():
    def __init__(self, channels, img_rows, img_cols, bs, nz, num_classes, data="mnist", data_path="/home/ubuntu/data/"):

        self.bs = bs
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.nz = nz
        self.num_classes = num_classes
        self.data = data

        if (self.data == "mnist") or (self.data == "fashion"):
            self.G = cdcgan_generator_mnist(self.channels, self.nz, self.num_classes)
            self.D = cdcgan_discriminator_mnist(self.channels, self.num_classes, self.img_rows)

            if self.data == "mnist":
                dataset = dset.MNIST(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_rows),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
            else:
                dataset = dset.FashionMNIST(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_rows),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

        if self.data == "cifar":
            self.G = cdcgan_generator_cifar(self.channels, self.nz, self.num_classes)
            self.D = cdcgan_discriminator_cifar(self.channels, self.num_classes, self.img_rows)

            dataset = dset.CIFAR10(root=data_path, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(img_rows),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


        self.dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        self.criterion = nn.BCELoss().cuda()

        self.optD = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.99))
        self.optG = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.99))

        for l in [self.G, self.D]:
            l.cuda()
#             l.apply(self.weights_init)

        self.train_hist = {}
        self.train_hist['d_loss'] = []
        self.train_hist['g_loss'] = []
        self.train_hist['epoch_time'] = []
        self.train_hist['total_time'] = []


    def to_categorical(self, y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    def create_fixed_inputs(self):
        # For saving imgs
        fixed_noise = torch.Tensor(100, self.nz).normal_(0, 1)

        np_fixed_c = np.arange(0, self.num_classes, 1)
        fixed_c = torch.from_numpy(np.repeat(np_fixed_c, 10, axis=0)).float()

        fixed_noise = Variable(fixed_noise.view(-1, self.nz, 1, 1).cuda(), requires_grad=False)
        fixed_c = Variable(fixed_c.cuda(), requires_grad=False)

        return fixed_noise, fixed_c


    def initialize_variables(self):
        real_x = torch.FloatTensor(self.bs, self.channels, self.img_rows, self.img_cols).cuda()
        real_c = torch.FloatTensor(self.bs).cuda()
        label = torch.FloatTensor(self.bs).cuda()
        noise = torch.FloatTensor(self.bs, self.nz).cuda()
        self.real_x = Variable(real_x)
        self.real_c = Variable(real_c)
        self.label = Variable(label, requires_grad=False)
        self.noise = Variable(noise)


    def batch_resize(self, x):
        batch = x.size(0)
        self.real_x.data.resize_(x.size())
        self.real_c.data.resize_(batch)
        self.label.data.resize_(batch)
        self.noise.data.resize_(batch, self.nz)


    def update_train_hist(self, d_loss, g_loss):
        self.train_hist['d_loss'].append(d_loss.data[0])
        self.train_hist['g_loss'].append(g_loss.data[0])


    def save_imgs(self, f_noise, f_c, epoch):
        to_save = self.G(f_noise, f_c)
        img_path = '/home/ubuntu/photos/pytorch-cdcgan-'+str(self.data)+'/%d.png' % (epoch)
        vutils.save_image(to_save.data, img_path, nrow=10, normalize=True)

    def plot_loss(self, path):
        x = range(len(self.train_hist['d_loss']))
        d_loss_hist = self.train_hist['d_loss']
        g_loss_hist = self.train_hist['g_loss']
        plt.plot(x, d_loss_hist, label='d_loss')
        plt.plot(x, g_loss_hist, label='g_loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.savefig(path)
        plt.close()


    def train(self, epochs=100, save_model=False, plot_loss=False):

        self.initialize_variables()

        fixed_noise, fixed_c = self.create_fixed_inputs()

        start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            for num_iters, batch_data in enumerate(self.dataloader, 0):

                # DISCRIMINATOR
                self.optD.zero_grad()
                x, c = batch_data

                self.batch_resize(x)

                self.real_x.data.copy_(x)
                self.real_c.data.copy_(c)
                pred_real = self.D(self.real_x, self.real_c)
                self.label.data.fill_(1)
                d_loss_real = self.criterion(pred_real, self.label)
                d_loss_real.backward()

                self.noise.data.normal_(0, 1)
                self.noise = self.noise.view(-1, self.nz, 1, 1)
                fake = self.G(self.noise, self.real_c)
                pred_fake = self.D(fake.detach(), self.real_c)     # dont train generator
                self.label.data.fill_(0)
                d_loss_fake = self.criterion(pred_fake, self.label)
                d_loss_fake.backward()

                d_loss_total = d_loss_real + d_loss_fake

                self.optD.step()


                # GENERATOR
                self.optG.zero_grad()
                pred_g = self.D(fake, self.real_c)
                self.label.data.fill_(1)

                g_loss = self.criterion(pred_g, self.label)

                g_loss.backward()
                self.optG.step()

                if plot_loss:
                    self.update_train_hist(d_loss_total, g_loss)

                if num_iters % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                            epoch, num_iters, d_loss_total.data.cpu().numpy(),
                            g_loss.data.cpu().numpy())
                         )

            self.train_hist['epoch_time'].append(time.time() - epoch_start_time)
            self.save_imgs(fixed_noise, fixed_c, epoch)


        print("Training is complete!")
        if save_model:
            path = "/home/ubuntu/saved_models/pytorch-cdcgan-" + str(self.data)
            print("Saving Model...")
            torch.save(self.G.state_dict(), path + "g.pth")
            torch.save(self.D.state_dict(), path + "d.pth")

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Total training time (%d epochs): " % epochs + str(self.train_hist['total_time'][0]))
        avg_epoch_time = np.mean(self.train_hist['epoch_time'])
        print("Average time for each epoch: %.2f" % avg_epoch_time)

        if plot_loss:
            self.plot_loss("/home/ubuntu/loss-plots/pytorch-cdcgan-" + str(self.data) + ".png")





class ACGAN():
    def __init__(self, channels, img_rows, img_cols, bs, nz, num_classes, data="mnist"):

        self.bs = bs
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.nz = nz
        self.num_classes = num_classes
        self.data = data

        if (self.data == "mnist") or (self.data == "fashion"):
            self.G = acgan_generator_mnist(self.channels, self.nz, self.num_classes)
            self.D = acgan_discriminator_mnist(self.channels, self.num_classes)

            if self.data == "mnist":
                dataset = dset.MNIST(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_rows),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
            else:
                dataset = dset.FashionMNIST(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_rows),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

        if self.data == "cifar":
            self.G = acgan_generator_cifar(self.channels, self.nz, self.num_classes)
            self.D = acgan_discriminator_cifar(self.channels, self.num_classes)

            dataset = dset.CIFAR10(root=data_path, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(img_rows),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


        self.dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        self.criterion_dis = nn.BCELoss().cuda()
        self.criterion_aux = nn.CrossEntropyLoss().cuda()

        self.optD = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.99))
        self.optG = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.99))

        for l in [self.G, self.D]:
            l.cuda()
            l.apply(self.weights_init)

        self.train_hist = {}
        self.train_hist['d_loss'] = []
        self.train_hist['g_loss'] = []
        self.train_hist['epoch_time'] = []
        self.train_hist['total_time'] = []


    def to_categorical(self, y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    def create_fixed_inputs(self):
        # For saving imgs
        fixed_noise = torch.Tensor(self.num_classes*10, self.nz).normal_(0, 1)

        np_fixed_c = self.to_categorical(np.arange(0, self.num_classes, 1), self.num_classes)
        fixed_c = torch.from_numpy(np.repeat(np_fixed_c, 10, axis=0)).float()

        fixed_noise = Variable(fixed_noise.view(-1, self.nz).cuda(), requires_grad=False)
        fixed_c = Variable(fixed_c.view(-1, self.num_classes).cuda(), requires_grad=False)

        return fixed_noise, fixed_c


    def initialize_variables(self):
        real_x = torch.FloatTensor(self.bs, self.channels, self.img_rows, self.img_cols).cuda()
        real_c = torch.LongTensor(self.bs).cuda()
        fake_c = torch.LongTensor(self.bs).cuda()
        fake_c_one_hot = torch.FloatTensor(self.bs, self.num_classes).cuda()
        label = torch.FloatTensor(self.bs).cuda()
        noise = torch.FloatTensor(self.bs, self.nz).cuda()
        self.real_x = Variable(real_x)
        self.real_c = Variable(real_c)
        self.fake_c = Variable(fake_c)
        self.fake_c_one_hot = Variable(fake_c_one_hot)
        self.label = Variable(label, requires_grad=False)
        self.noise = Variable(noise)

    def batch_resize(self, x):
        batch = x.size(0)
        self.real_x.data.resize_(x.size())
        self.real_c.data.resize_(batch)
        self.fake_c.data.resize_(batch)
        self.fake_c_one_hot.data.resize_(batch, self.num_classes)
        self.label.data.resize_(batch)
        self.noise.data.resize_(batch, self.nz)

    def update_train_hist(self, d_loss, g_loss):
        self.train_hist['d_loss'].append(d_loss.data[0])
        self.train_hist['g_loss'].append(g_loss.data[0])


    def save_imgs(self, f_noise, f_c, epoch):
        to_save = self.G(f_noise, f_c)
        img_path = '/home/ubuntu/photos/pytorch-acgan-'+str(self.data)+'/%d.png' % (epoch)
        vutils.save_image(to_save.data, img_path, nrow=self.num_classes, normalize=True)

    def plot_loss(self, path):
        x = range(len(self.train_hist['d_loss']))
        d_loss_hist = self.train_hist['d_loss']
        g_loss_hist = self.train_hist['g_loss']
        plt.plot(x, d_loss_hist, label='d_loss')
        plt.plot(x, g_loss_hist, label='g_loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.savefig(path)
        plt.close()


    def train(self, epochs=100, save_model=False, plot_loss=False):

        self.initialize_variables()

        fixed_noise, fixed_c = self.create_fixed_inputs()

        start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            for num_iters, batch_data in enumerate(self.dataloader, 0):

                # DISCRIMINATOR
                self.optD.zero_grad()
                x, c = batch_data

                # To accommodate batches of size < self.bs
                self.batch_resize(x)

                self.real_x.data.copy_(x)
                self.real_c.data.copy_(c)
                pred_real, pred_aux_real = self.D(self.real_x)
                self.label.data.fill_(1)
                d_loss_dis_real = self.criterion_dis(pred_real, self.label)
                d_loss_aux_real = self.criterion_aux(pred_aux_real, self.real_c) * 0.5
                d_loss_real = d_loss_dis_real + d_loss_aux_real
                d_loss_real.backward()

                self.noise.data.normal_(0, 1)
                self.label.data.fill_(0)
                np_fake_c = np.random.randint(0, self.num_classes, x.size(0))
                self.fake_c.data.copy_(torch.from_numpy(np_fake_c))
                self.fake_c_one_hot.data.copy_(torch.from_numpy(self.to_categorical(np_fake_c, self.num_classes)))
                fake_img = self.G(self.noise, self.fake_c_one_hot)
                pred_fake, pred_aux_fake = self.D(fake_img.detach())
                d_loss_dis_fake = self.criterion_dis(pred_fake, self.label)
                d_loss_aux_fake = self.criterion_aux(pred_aux_fake, self.fake_c) * 0.5
                d_loss_fake = d_loss_dis_fake + d_loss_aux_fake
                d_loss_fake.backward()
                d_loss_total = d_loss_real + d_loss_fake
                self.optD.step()


                # GENERATOR
                self.optG.zero_grad()
                pred_g, pred_aux_g = self.D(fake_img)
                self.label.data.fill_(1)
                g_loss_dis = self.criterion_dis(pred_g, self.label)
                g_loss_aux = self.criterion_aux(pred_aux_g, self.fake_c)
                g_loss = g_loss_dis + g_loss_aux

                g_loss.backward()
                self.optG.step()

                if plot_loss:
                    self.update_train_hist(d_loss_total, g_loss)

                if num_iters % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                            epoch, num_iters, d_loss_total.data.cpu().numpy(),
                            g_loss.data.cpu().numpy())
                         )

            self.train_hist['epoch_time'].append(time.time() - epoch_start_time)
            self.save_imgs(fixed_noise, fixed_c, epoch)


        print("Training is complete!")
        if save_model:
            path = "/home/ubuntu/saved_models/pytorch-acgan-" + str(self.data)
            print("Saving Model...")
            torch.save(self.G.state_dict(), path + "g.pth")
            torch.save(self.D.state_dict(), path + "d.pth")

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Total training time (%d epochs): " % epochs + str(self.train_hist['total_time'][0]))
        avg_epoch_time = np.mean(self.train_hist['epoch_time'])
        print("Average time for each epoch: %.2f" % avg_epoch_time)

        if plot_loss:
            self.plot_loss("/home/ubuntu/loss-plots/pytorch-acgan-" + str(self.data) + ".png")






class TwoNGAN():
    def __init__(self, channels, img_rows, img_cols, bs, nz, num_classes, data="mnist", data_path="/home/ubuntu/data/"):

        self.bs = bs
        self.channels = channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.nz = nz
        self.num_classes = num_classes
        self.data = data

        if (self.data == "mnist") or (self.data == "fashion"):
            self.G = twoNgan_generator_mnist(self.channels, self.nz, self.num_classes)
            self.D = twoNgan_discriminator_mnist(self.channels, self.num_classes)

            if self.data == "mnist":
                dataset = dset.MNIST(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_rows),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
            else:
                dataset = dset.FashionMNIST(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_rows),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

        if self.data == "cifar":
            self.G = twoNgan_generator_cifar(self.channels, self.nz, self.num_classes)
            self.D = twoNgan_discriminator_cifar(self.channels, self.num_classes)

            dataset = dset.CIFAR10(root=data_path, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(img_rows),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


        self.dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optD = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.99))
        self.optG = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.99))

        for l in [self.G, self.D]:
            l.cuda()
            l.apply(self.weights_init)

        self.train_hist = {}
        self.train_hist['d_loss'] = []
        self.train_hist['g_loss'] = []
        self.train_hist['epoch_time'] = []
        self.train_hist['total_time'] = []


    def to_categorical(self, y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    def create_fixed_inputs(self):
        # For saving imgs
        fixed_noise = torch.Tensor(self.num_classes*10, self.nz).normal_(0, 1)

        np_fixed_c = np.arange(0, self.num_classes, 1)
        fixed_c = torch.from_numpy(np.repeat(np_fixed_c, 10, axis=0)).float()

        fixed_noise = Variable(fixed_noise.view(-1, self.nz).cuda(), requires_grad=False)
        fixed_c = Variable(fixed_c.view(-1).long().cuda(), requires_grad=False)

        return fixed_noise, fixed_c


    def initialize_variables(self):
        real_x = torch.FloatTensor(self.bs, self.channels, self.img_rows, self.img_cols).cuda()
        real_c = torch.LongTensor(self.bs).cuda()
        fake_c = torch.LongTensor(self.bs).cuda()
        fake_c_for_d = torch.LongTensor(self.bs).cuda()
        noise = torch.FloatTensor(self.bs, self.nz).cuda()
        self.real_x = Variable(real_x)
        self.real_c = Variable(real_c)
        self.fake_c = Variable(fake_c)
        self.fake_c_for_d = Variable(fake_c_for_d)
        self.noise = Variable(noise)

    def batch_resize(self, x):
        batch = x.size(0)
        self.real_x.data.resize_(x.size())
        self.real_c.data.resize_(batch)
        self.fake_c.data.resize_(batch)
        self.fake_c_for_d.data.resize_(batch)
        self.noise.data.resize_(batch, self.nz)

    def update_train_hist(self, d_loss, g_loss):
        self.train_hist['d_loss'].append(d_loss.data[0])
        self.train_hist['g_loss'].append(g_loss.data[0])


    def save_imgs(self, f_noise, f_c, epoch):
        to_save = self.G(f_noise, f_c)
        img_path = '/home/ubuntu/photos/pytorch-twoNgan-'+str(self.data)+'/%d.png' % (epoch)
        vutils.save_image(to_save.data, img_path, nrow=self.num_classes, normalize=True)

    def plot_loss(self, path):
        x = range(len(self.train_hist['d_loss']))
        d_loss_hist = self.train_hist['d_loss']
        g_loss_hist = self.train_hist['g_loss']
        plt.plot(x, d_loss_hist, label='d_loss')
        plt.plot(x, g_loss_hist, label='g_loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend(loc=4)
        plt.savefig(path)
        plt.close()


    def train(self, epochs=100, save_model=False, plot_loss=False):

        self.initialize_variables()

        fixed_noise, fixed_c = self.create_fixed_inputs()

        start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            for num_iters, batch_data in enumerate(self.dataloader, 0):

                # DISCRIMINATOR
                self.optD.zero_grad()
                x, c = batch_data

                # To accommodate batches of size < self.bs
                self.batch_resize(x)

                self.real_x.data.copy_(x)
                self.real_c.data.copy_(c)
                pred_real = self.D(self.real_x)
                d_loss_real = self.criterion(pred_real, self.real_c)
                d_loss_real.backward()

                self.noise.data.normal_(0, 1)
                np_fake_c = np.random.randint(0, self.num_classes, x.size(0))
                self.fake_c.data.copy_(torch.from_numpy(np_fake_c))
                self.fake_c_for_d.data.copy_(torch.from_numpy(np_fake_c+self.num_classes))
                fake_img = self.G(self.noise, self.fake_c)
                pred_fake = self.D(fake_img.detach())
                d_loss_fake = self.criterion(pred_fake, self.fake_c_for_d)
                d_loss_fake.backward()
                d_loss_total = d_loss_real + d_loss_fake
                self.optD.step()


                # GENERATOR
                self.optG.zero_grad()
                pred_g = self.D(fake_img)
                g_loss = self.criterion(pred_g, self.fake_c)

                g_loss.backward()
                self.optG.step()

                if plot_loss:
                    self.update_train_hist(d_loss_total, g_loss)

                if num_iters % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                            epoch, num_iters, d_loss_total.data.cpu().numpy(),
                            g_loss.data.cpu().numpy())
                         )

            self.train_hist['epoch_time'].append(time.time() - epoch_start_time)
            self.save_imgs(fixed_noise, fixed_c, epoch)


        print("Training is complete!")
        if save_model:
            path = "/home/ubuntu/saved_models/pytorch-twoNgan-" + str(self.data)
            print("Saving Model...")
            torch.save(self.G.state_dict(), path + "g.pth")
            torch.save(self.D.state_dict(), path + "d.pth")

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Total training time (%d epochs): " % epochs + str(self.train_hist['total_time'][0]))
        avg_epoch_time = np.mean(self.train_hist['epoch_time'])
        print("Average time for each epoch: %.2f" % avg_epoch_time)

        if plot_loss:
            self.plot_loss("/home/ubuntu/loss-plots/pytorch-twoNgan-" + str(self.data) + ".png")
