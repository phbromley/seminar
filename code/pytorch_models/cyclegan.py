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
import random

import numpy as np
from matplotlib import pyplot as plt


# Example training code:
# IMG_ROWS = 256
# IMG_COLS = 256
# CHANNELS_A = 3
# CHANNELS_B = 3
# BATCH_SIZE = 1
# N_BLOCKS = 9
# USE_MSE = True
# CYCLE_LAMBDA = 10.0
# IDT_LAMBDA = 0.5
#
# cyclegan = CycleGAN(channels_A=CHANNELS_A, channels_B=CHANNELS_B, img_rows=IMG_ROWS, img_cols=IMG_COLS,
#                     bs=BATCH_SIZE, n_blocks=N_BLOCKS, use_mse=USE_MSE, cycle_lambda=CYCLE_LAMBDA,
#                     idt_lambda=IDT_LAMBDA, data="monet2photo")


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class resnet_block(nn.Module):
    def __init__(self, channels):
        super(resnet_block, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, input):
        residual = input
        block = self.net(input)
        out = residual + block
        return out


class cyclegan_generator(nn.Module):
    def __init__(self, rows, cols, channels_A, channels_B, n_blocks):
        super(cyclegan_generator, self).__init__()                       # (assuming 256*256 imgs...)
        self.encode = [nn.Conv2d(channels_A, 64, 7, 1, 3, bias=False),     # 64*256*256
                       nn.InstanceNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(64, 128, 3, 2, 1, bias=False),          # 128*128*128
                       nn.InstanceNorm2d(128),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(128, 256, 3, 2, 1, bias=False),         # 256*64*64
                       nn.InstanceNorm2d(256),
                       nn.ReLU(inplace=True)]

        self.resnet = []
        for i in range(n_blocks):
            self.resnet += [resnet_block(256)]

        self.decode = [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1, bias=False), # 128*128*128
                       nn.InstanceNorm2d(128),
                       nn.ReLU(inplace=True),
                       nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1, bias=False),  # 64*256*256
                       nn.InstanceNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(64, channels_B, 7, 1, 3, bias=False)]      # channels_B*256*256

        self.full_net = nn.Sequential(*(self.encode+self.resnet+self.decode))

    def forward(self, input):
        output = self.full_net(input)
        return output


# "PatchGAN" is just a convnet that outputs a *grid* of predictions of whether or not a *patch* is real
class cyclegan_discriminator(nn.Module):
    def __init__(self, channels):
        super(cyclegan_discriminator, self).__init__()

        self.patchgan = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),  # 64*128*128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),       # 128*64*64
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),      # 256*32*32
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),      # 512*31*31
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),        # 1*30*30  (patchgan output grid)
            nn.Sigmoid()
        )


    def forward(self, input):
        output = self.patchgan(input)
        return output


# Taken directly from official cyclegan PyTorch implementation github repo:
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images



# Much inspiration taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class CycleGAN():
    def __init__(self, channels_A, channels_B, img_rows, img_cols, bs, n_blocks,
                 use_mse, cycle_lambda, idt_lambda, data="monet2photo"):
        self.channels_A = channels_A
        self.channels_B = channels_B
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.bs = bs
        self.n_blocks = n_blocks
        self.data = data
        self.use_mse = use_mse
        self.cycle_lambda = cycle_lambda
        self.idt_lambda = idt_lambda

        self.G_AB = cyclegan_generator(self.img_rows, self.img_cols,
                                       self.channels_A, self.channels_B, self.n_blocks)

        self.G_BA = cyclegan_generator(self.img_rows, self.img_cols,
                                       self.channels_B, self.channels_A, self.n_blocks)

        self.D_A = cyclegan_discriminator(self.channels_A)

        self.D_B = cyclegan_discriminator(self.channels_B)

        if self.data == "monet2photo":
            self.data_A = dset.ImageFolder(root='/home/ubuntu/data/monet2photo/trainA/',
                                           transform=transforms.Compose([
                                            transforms.Resize(self.img_rows),
                                            transforms.CenterCrop(self.img_cols),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
            self.data_B = dset.ImageFolder(root='/home/ubuntu/data/monet2photo/trainB/',
                                           transform=transforms.Compose([
                                            transforms.Resize(self.img_rows),
                                            transforms.CenterCrop(self.img_cols),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

        if self.data == "face2cartoon":
            self.data_A = dset.ImageFolder(root='/home/ubuntu/data/lfw-deepfunneled/',
                                           transform=transforms.Compose([
                                            transforms.Resize(self.img_rows),
                                            transforms.CenterCrop(self.img_cols),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
            self.data_B = dset.ImageFolder(root='/home/ubuntu/data/cartoons/',
                                           transform=transforms.Compose([
                                            transforms.Resize(self.img_rows),
                                            transforms.CenterCrop(self.img_cols),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

        if self.data == "nicecar2normalcar":
            self.data_A = dset.ImageFolder(root='/home/ubuntu/data/cars_train/nice/',
                                           transform=transforms.Compose([
                                            transforms.Resize(self.img_rows),
                                            transforms.CenterCrop(self.img_cols),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
            self.data_B = dset.ImageFolder(root='/home/ubuntu/data/cars_train/not/',
                                           transform=transforms.Compose([
                                            transforms.Resize(self.img_rows),
                                            transforms.CenterCrop(self.img_cols),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

        self.dataloader = torch.utils.data.DataLoader(ConcatDataset(self.data_A, self.data_B),
                                                      batch_size=self.bs, shuffle=True,
                                                      num_workers=4)

        self.criterionBCE = nn.BCELoss().cuda()
        self.criterionMSE = nn.BCELoss().cuda()
        self.criterionCycle = nn.L1Loss().cuda()
        self.criterionIdt = nn.L1Loss().cuda()

        self.optD = optim.Adam([{'params':self.D_A.parameters()},
                                {'params':self.D_B.parameters()}], lr=0.0002, betas=(0.5, 0.99))
        self.optG = optim.Adam([{'params':self.G_AB.parameters()},
                                {'params':self.G_BA.parameters()}], lr=0.0002, betas=(0.5, 0.99))

        for l in [self.G_AB, self.G_BA, self.D_A, self.D_B]:
            l.cuda()
            l.apply(self.weights_init)

        self.train_hist = {}
        self.train_hist['d_a_loss'] = []
        self.train_hist['d_b_loss'] = []
        self.train_hist['g_ab_loss'] = []
        self.train_hist['g_ba_loss'] = []
        self.train_hist['cycle_A_loss'] = []
        self.train_hist['cycle_B_loss'] = []
        self.train_hist['epoch_time'] = []
        self.train_hist['total_time'] = []


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    # Helper function to deal with PatchGAN discriminator
    def calc_d_g_loss(self, preds, is_real, use_mse=True):
        if is_real:
            target = torch.FloatTensor(preds.size()).fill_(1.0).cuda()
            target = Variable(target, requires_grad=False)
        else:
            target = torch.FloatTensor(preds.size()).fill_(0.0).cuda()
            target = Variable(target, requires_grad=False)

        if use_mse:
            return self.criterionMSE(preds, target)
        else:
            return self.criterionBCE(preds, target)


    def get_fixed_data(self, num_imgs):
        fixed_A = torch.FloatTensor(num_imgs, self.channels_A, self.img_rows, self.img_cols)
        for i in range(num_imgs):
            fixed_A[i] = self.data_A[i][0]

        fixed_B = torch.FloatTensor(num_imgs, self.channels_B, self.img_rows, self.img_cols)
        for i in range(num_imgs):
            fixed_B[i] = self.data_B[i][0]

        return fixed_A, fixed_B


    def update_train_hist(self, d_a_loss, d_b_loss, g_ab_loss, g_ba_loss, cycle_A_loss, cycle_B_loss):
        self.train_hist['d_a_loss'].append(d_a_loss.data[0])
        self.train_hist['d_b_loss'].append(d_b_loss.data[0])
        self.train_hist['g_ab_loss'].append(g_ab_loss.data[0])
        self.train_hist['g_ba_loss'].append(g_ba_loss.data[0])
        self.train_hist['cycle_A_loss'].append(cycle_A_loss.data[0])
        self.train_hist['cycle_B_loss'].append(cycle_B_loss.data[0])



    def plot_loss(self, path):
        x = range(len(self.train_hist['d_a_loss']))
        d_a_loss_hist = self.train_hist['d_a_loss']
        d_b_loss_hist = self.train_hist['d_b_loss']
        g_ab_loss_hist = self.train_hist['g_ab_loss']
        g_ba_loss_hist = self.train_hist['g_ba_loss']
        cycle_A_loss_hist = self.train_hist['cycle_A_loss']
        cycle_B_loss_hist = self.train_hist['cycle_B_loss']
        plt.plot(x, d_a_loss_hist, label='d_a_loss')
        plt.plot(x, d_b_loss_hist, label='d_b_loss')
        plt.plot(x, g_ab_loss_hist, label='g_ab_loss')
        plt.plot(x, g_ba_loss_hist, label='g_ba_loss')
        plt.plot(x, cycle_A_loss_hist, label='cycle_A_loss')
        plt.plot(x, cycle_B_loss_hist, label='cycle_B_loss')
        plt.legend(loc=4)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.savefig(path)
        plt.close()


    def train(self, epochs=50, save_model=True, plot_loss=True):

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        real_A = torch.FloatTensor(self.bs, self.channels_A, self.img_rows, self.img_cols).cuda()
        real_B = torch.FloatTensor(self.bs, self.channels_B, self.img_rows, self.img_cols).cuda()
        fake_A = torch.FloatTensor(self.bs, self.channels_A, self.img_rows, self.img_cols).cuda()
        fake_B = torch.FloatTensor(self.bs, self.channels_B, self.img_rows, self.img_cols).cuda()

        real_A = Variable(real_A)
        real_B = Variable(real_B)
        fake_A = Variable(fake_A)
        fake_B = Variable(fake_B)

        fake_A_pool = ImagePool(50)
        fake_B_pool = ImagePool(50)

        fixed_A, fixed_B = self.get_fixed_data(9)
        fixed_A = Variable(fixed_A.cuda(), volatile=True)
        fixed_B = Variable(fixed_B.cuda(), volatile=True)


        start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            for i, (batch_A, batch_B) in enumerate(self.dataloader):

                # Load the data into variables
                x_A, _ = batch_A
                x_B, _ = batch_B
                real_A.data.copy_(x_A)
                real_B.data.copy_(x_B)

                ###########################
                ## UPDATE DISCRIMINATORS ##
                ###########################

                self.optD.zero_grad()

                # Update D_A (takes real_A and fake_A and classifies real/fake)
                pred_real_A = self.D_A(real_A)
                d_A_loss_real = self.calc_d_g_loss(pred_real_A, True, use_mse=self.use_mse)

                generated_A_from_B = self.G_BA(real_B)
                generated_A_from_B = fake_A_pool.query(generated_A_from_B)
                pred_fake_A = self.D_A(generated_A_from_B.detach())
                d_A_loss_fake = self.calc_d_g_loss(pred_fake_A, False, use_mse=self.use_mse)

                loss_d_A = (d_A_loss_real + d_A_loss_fake) * 0.5
                loss_d_A.backward()


                # Update D_B (takes real_B and fake_B and classifies real/fake)
                pred_real_B = self.D_B(real_B)
                d_B_loss_real = self.calc_d_g_loss(pred_real_B, True, use_mse=self.use_mse)

                generated_B_from_A = self.G_AB(real_A)
                generated_B_from_A = fake_B_pool.query(generated_B_from_A)
                pred_fake_B = self.D_B(generated_B_from_A.detach())
                d_B_loss_fake = self.calc_d_g_loss(pred_fake_B, False, use_mse=self.use_mse)

                loss_d_B = (d_B_loss_real + d_B_loss_fake) * 0.5
                loss_d_B.backward()

                self.optD.step()


                #######################
                ## UPDATE GENERATORS ##
                #######################

                self.optG.zero_grad()

                # Update G_AB (takes real_A, translates to fake_B, maximize D_B(G_AB(real_A)))
                B_from_A = self.G_AB(real_A)
                pred_fake_B_G = self.D_B(B_from_A)
                g_AB_loss = self.calc_d_g_loss(pred_fake_B_G, True, use_mse=self.use_mse)

                # Update G_BA (takes real_B, translates to fake_A, maximize D_A(G_BA(real_B)))
                A_from_B = self.G_BA(real_B)
                pred_fake_A_G = self.D_A(A_from_B)
                g_BA_loss = self.calc_d_g_loss(pred_fake_A_G, True, use_mse=self.use_mse)

                # Update cycle loss A (minimize dist btw real A, re-translated A)
                A_back_from_B = self.G_BA(B_from_A)
                cycleA_loss = self.criterionCycle(A_back_from_B, real_A) * self.cycle_lambda

                # Update cycle loss B (minimize dist btw real B, re-translated B)
                B_back_from_A = self.G_AB(A_from_B)
                cycleB_loss = self.criterionCycle(B_back_from_A, real_B) * self.cycle_lambda


                # Update identity loss for paintings (preserves color by regularizing the generator
                #   to map real images of the TARGET domain using the identity function. I.e. we want
                #   G_AB(real_B) to be real_B)
                # For some reason CycleGAN code (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
                # multiplies identity loss by both cycle lambda and idt lambda

                # Identity A
                A_to_A = self.G_BA(real_A)
                identityA_loss = self.criterionIdt(A_to_A, real_A) * self.cycle_lambda * self.idt_lambda

                # Identity B
                B_to_B = self.G_AB(real_B)
                identityB_loss = self.criterionIdt(B_to_B, real_B) * self.cycle_lambda * self.idt_lambda

                loss_g_total = g_AB_loss + g_BA_loss + cycleA_loss + cycleB_loss + identityA_loss + identityB_loss
                loss_g_total.backward()
                self.optG.step()

                if plot_loss:
                    self.update_train_hist(loss_d_A, loss_d_B, g_AB_loss, g_BA_loss, cycleA_loss, cycleB_loss)



                if i % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, D_A loss: {2}, D_B loss: {3}, G loss: {4}'.format(
                            epoch, i, loss_d_A.data.cpu().numpy(), loss_d_B.data.cpu().numpy(),
                            loss_g_total.data.cpu().numpy())
                          )
                    to_save_fake_B = self.G_AB(fixed_A)
                    to_save_fake_A = self.G_BA(fixed_B)
                    to_save_cycle_A = self.G_BA(to_save_fake_B)
                    to_save_cycle_B = self.G_AB(to_save_fake_A)


                    # normalize=True important! Otherwise all images look dark
                    img_path_fake_B = '/home/ubuntu/photos/pytorch-cyclegan-nicecar2normalcar/%d_%d_B.png' % (epoch, i)
                    img_path_fake_A = '/home/ubuntu/photos/pytorch-cyclegan-nicecar2normalcar/%d_%d_A.png' % (epoch, i)
                    img_path_cycle_B = '/home/ubuntu/photos/pytorch-cyclegan-nicecar2normalcar/%d_%d_cycleB.png' % (epoch, i)
                    img_path_cycle_A = '/home/ubuntu/photos/pytorch-cyclegan-nicecar2normalcar/%d_%d_cycleA.png' % (epoch, i)
                    vutils.save_image(to_save_fake_B.data, img_path_fake_B, nrow=3, normalize=True)
                    vutils.save_image(to_save_fake_A.data, img_path_fake_A, nrow=3, normalize=True)
                    vutils.save_image(to_save_cycle_B.data, img_path_cycle_B, nrow=3, normalize=True)
                    vutils.save_image(to_save_cycle_A.data, img_path_cycle_A, nrow=3, normalize=True)


#                 del loss_g_total, loss_d_A, loss_d_B, pred_real_A, pred_fake_A, pred_real_B, pred_fake_B
#                 del pred_fake_B_G, pred_fake_A_G, A_back_from_B, B_back_from_A, A_to_A, B_to_B

            self.train_hist['epoch_time'].append(time.time() - epoch_start_time)
            print("Epoch Time: " + str(time.time()-epoch_start_time))
            if epoch % 10 == 0:
                if plot_loss:
                    self.plot_loss("/home/ubuntu/loss-plots/pytorch-cyclegan-nicecar2normalcar_%d.png" % epoch)



        print("Training is complete!")
        if save_model:
            path = "/home/ubuntu/saved_models/pytorch-cyclegan-nicecar2normalcar"
            torch.save(self.G_AB.state_dict(), path + "G_AB.pth")
            torch.save(self.G_BA.state_dict(), path + "G_BA.pth")
            torch.save(self.D_A.state_dict(), path + "D_A.pth")
            torch.save(self.D_B.state_dict(), path + "D_B.pth")


        self.train_hist['total_time'].append(time.time() - start_time)
        print("Total training time (%d epochs): " % epochs + str(self.train_hist['total_time'][0]))
        avg_epoch_time = np.mean(self.train_hist['epoch_time'])
        print("Average time for each epoch: %.2f" % avg_epoch_time)

        if plot_loss:
            self.plot_loss("/home/ubuntu/loss-plots/pytorch-cyclegan-nicecar2normalcar.png")
