import torch
import torch.nn as nn
import torch.nn.parallel


class cdcgan_generator_cifar(nn.Module):
    def __init__(self, channels, nz, num_classes):
        super(cdcgan_generator_cifar, self).__init__()
        self.num_classes = num_classes

        self.embed = nn.Embedding(num_classes, num_classes)

        self.noise_conv = nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias=False)  #makes size 256, 4, 4
        self.noise_bn = nn.BatchNorm2d(256)
        self.noise_relu = nn.ReLU()

        self.label_conv = nn.ConvTranspose2d(num_classes, 256, 4, 1, 0, bias=False) #makes size 256, 4, 4
        self.label_bn = nn.BatchNorm2d(256)
        self.label_relu = nn.ReLU()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # forward pass
    def forward(self, input, label):
        z = self.noise_conv(input)
        z = self.noise_bn(z)
        z = self.noise_relu(z)
        embed = self.embed(label).view(-1, self.num_classes, 1, 1).float()
        c = self.label_conv(label)
        c = self.label_bn(c)
        c = self.label_relu(c)
        cat = torch.cat([z, c], 1)   # size is now 512, 4, 4
        output = self.net(cat)
        return output


class cdcgan_discriminator_cifar(nn.Module):
    def __init__(self, channels, num_classes, img_rows):
        super(cdcgan_discriminator_cifar, self).__init__()
        self.num_classes = num_classes

        self.embed = nn.Embedding(num_classes, num_classes)

        self.label_conv = nn.ConvTranspose2d(num_classes, 1, img_rows, 2, 0, bias=False)
        self.label_bn = nn.BatchNorm2d(1)
        self.label_relu = nn.LeakyReLU(0.2, inplace=True)

        self.net = nn.Sequential(
            nn.Conv2d(channels+1, 64, 4, 2, 1, bias=False),
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

    def forward(self, input, label):
        embed = self.embed(label).view(-1, self.num_classes, 1, 1).float()
        c = self.label_conv(embed)
        c = self.label_bn(c)
        c = self.label_relu(c)
        cat = torch.cat([input, c], 1)
        output = self.net(cat).view(-1, 1)
        return output


class cdcgan_generator_mnist(nn.Module):
    def __init__(self, channels, nz, num_classes):
        super(cdcgan_generator_mnist, self).__init__()
        self.num_classes = num_classes

        self.embed = nn.Embedding(num_classes, num_classes)

        self.noise_conv = nn.ConvTranspose2d(nz, 128, 7, 1, 0, bias=False)
        self.noise_bn = nn.BatchNorm2d(128)
        self.noise_relu = nn.ReLU()

        self.label_conv = nn.ConvTranspose2d(num_classes, 128, 7, 1, 0, bias=False)
        self.label_bn = nn.BatchNorm2d(128)
        self.label_relu = nn.ReLU()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, input, label):
        z = self.noise_conv(input)
        z = self.noise_bn(z)
        z = self.noise_relu(z)
        embed = self.embed(label.long()).view(-1, self.num_classes, 1, 1)
        c = self.label_conv(embed)
        c = self.label_bn(c)
        c = self.label_relu(c)
        cat = torch.cat([z, c], 1)
        output = self.net(cat)
        return output


class cdcgan_discriminator_mnist(nn.Module):
    def __init__(self, channels, num_classes, img_rows):
        super(cdcgan_discriminator_mnist, self).__init__()
        self.num_classes = num_classes

        self.embed = nn.Embedding(num_classes, num_classes)

        self.label_conv = nn.ConvTranspose2d(num_classes, 1, img_rows, 2, 0, bias=False)
        self.label_bn = nn.BatchNorm2d(1)
        self.label_relu = nn.LeakyReLU(0.2, inplace=True)

        self.net = nn.Sequential(
            nn.Conv2d(channels+1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input, label):
        embed = self.embed(label.long()).view(-1, self.num_classes, 1, 1)
        c = self.label_conv(embed)
        c = self.label_bn(c)
        c = self.label_relu(c)
        cat = torch.cat([input, c], 1)
        output = self.net(cat).view(-1, 1)
        return output



class acgan_generator_cifar(nn.Module):
    def __init__(self, channels, nz, num_classes):
        super(acgan_generator_cifar, self).__init__()

        self.nz = nz
        self.num_classes = num_classes

        self.fc = nn.Linear(nz+num_classes, 384)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, noise, label):
        concat = torch.cat([noise, label], 1)
        linear = self.fc(concat)
        output = self.net(linear.view(-1, 384, 1, 1))
        return output


class acgan_discriminator_cifar(nn.Module):
    def __init__(self, channels, num_classes):
        super(acgan_discriminator_cifar, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

        )
        self.dense_sig = nn.Linear(512*4*4, 1)
        self.sigmoid = nn.Sigmoid()

        self.dense_aux = nn.Linear(512*4*4, num_classes)
        self.softmax = nn.Softmax()


    def forward(self, input):
        net = self.net(input)
        net = net.view(-1, 512*4*4)
        sig_out = self.dense_sig(net)
        sig_out = self.sigmoid(sig_out)
        aux_out = self.dense_aux(net)
        aux_out = self.softmax(aux_out)
        return sig_out.squeeze(), aux_out.squeeze()


class acgan_generator_mnist(nn.Module):
    def __init__(self, channels, nz, num_classes):
        super(acgan_generator_mnist, self).__init__()

        self.nz = nz
        self.num_classes = num_classes

        self.fc = nn.Linear(nz+num_classes, 384)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 7, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, noise, label):
        concat = torch.cat([noise, label], 1)
        linear = self.fc(concat)
        output = self.net(linear.view(-1, 384, 1, 1))
        return output


class acgan_discriminator_mnist(nn.Module):
    def __init__(self, channels, num_classes):
        super(acgan_discriminator_mnist, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )
        self.dense_sig = nn.Linear(512*4*4, 1)
        self.sigmoid = nn.Sigmoid()

        self.dense_aux = nn.Linear(512*4*4, num_classes)
        self.softmax = nn.Softmax()


    def forward(self, input):
        net = self.net(input)
        net = net.view(-1, 512*4*4)
        sig_out = self.dense_sig(net)
        sig_out = self.sigmoid(sig_out)
        aux_out = self.dense_aux(net)
        aux_out = self.softmax(aux_out)
        return sig_out.squeeze(), aux_out.squeeze()




class twoNgan_generator_cifar(nn.Module):
    def __init__(self, channels, nz, num_classes):
        super(twoNgan_generator_cifar, self).__init__()

        self.nz = nz
        self.num_classes = num_classes

        self.embed = nn.Embedding(num_classes, num_classes)

        self.fc = nn.Linear(nz+num_classes, 1024)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, noise, label):
        embed = self.embed(label)
        concat = torch.cat([noise, embed], 1)
        fc = self.fc(concat).view(-1, 1024, 1, 1)
        output = self.net(fc)
        return output


class twoNgan_discriminator_cifar(nn.Module):
    def __init__(self, channels, num_classes):
        super(twoNgan_discriminator_cifar, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(256, 1024, 4, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )

        self.dense = nn.Linear(1024, num_classes*2)
        self.softmax = nn.Softmax()


    def forward(self, input):
        net = self.net(input)
        fc = self.dense(net.squeeze())
        output = self.softmax(fc)
        return output


class twoNgan_generator_mnist(nn.Module):
    def __init__(self, channels, nz, num_classes):
        super(twoNgan_generator_mnist, self).__init__()

        self.embed = nn.Embedding(num_classes, num_classes)

        self.fc = nn.Linear(nz+num_classes, 384)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 7, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, noise, label):
        embed = self.embed(label)
        concat = torch.cat([noise, embed], 1)
        fc = self.fc(concat).view(-1, 384, 1, 1)
        output = self.net(fc)
        return output


class twoNgan_discriminator_mnist(nn.Module):
    def __init__(self, channels, num_classes):
        super(twoNgan_discriminator_mnist, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )

        self.dense = nn.Linear(1024, num_classes*2)
        self.softmax = nn.Softmax()


    def forward(self, input):
        net = self.net(input)
        fc = self.dense(net.squeeze())
        output = self.softmax(fc)
        return output




# For all infogan networks: Much inspiration taken from:
#    https://github.com/pianomania/infoGAN-pytorch/blob/master/trainer.py
class infogan_generator_mnist(nn.Module):
    def __init__(self, latent_size, channels):
        super(infogan_generator_mnist, self).__init__()
        self.latent_size = latent_size
        self.channels = channels

        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 1024, 1, 1, 0, bias=False),  #essentially fc layer
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, cat, cont):
        input = torch.cat([noise, cat, cont], 1).view(-1, self.latent_size, 1, 1)
        output = self.net(input)
        return output


class infogan_d_q_body_mnist(nn.Module):
    def __init__(self, channels):
        super(infogan_d_q_body_mnist, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )


    def forward(self, input):
        output = self.net(input)
        return output


class infogan_discriminator(nn.Module):
    def __init__(self):
        super(infogan_discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1024, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.net(input)
        return output.squeeze()


class infogan_q(nn.Module):
    def __init__(self, cat_size, cont_size):
        super(infogan_q, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1024, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.cat_out = nn.Conv2d(128, cat_size, 1, 1, 0)
        self.mean_cont_out = nn.Conv2d(128, cont_size, 1, 1, 0)
        self.var_cont_out = nn.Conv2d(128, cont_size, 1, 1, 0)

    def forward(self, input):
        q = self.net(input)
        cat = self.cat_out(q).squeeze()
        mean = self.mean_cont_out(q).squeeze()
        var = self.var_cont_out(q).squeeze().exp()
        return cat, mean, var


class infogan_generator_cifar(nn.Module):
    def __init__(self, latent_size, channels):
        super(infogan_generator_cifar, self).__init__()
        self.latent_size = latent_size
        self.channels = channels

        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 1024, 1, 1, 0, bias=False),  #essentially fc layer
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, cat, cont):
        input = torch.cat([noise, cat, cont], 1).view(-1, self.latent_size, 1, 1)
        output = self.net(input)
        return output


class infogan_d_q_body_cifar(nn.Module):
    def __init__(self, channels):
        super(infogan_d_q_body_cifar, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
            nn.Conv2d(256, 1024, 4, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25, inplace=False),
        )


    def forward(self, input):
        output = self.net(input)
        return output



class infogan_generator_cats(nn.Module):
    def __init__(self, latent_size, channels):
        super(infogan_generator_cats, self).__init__()
        self.latent_size = latent_size
        self.channels = channels

        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 1024, 1, 1, 0, bias=False),  #essentially fc layer
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, cat, cont):
        input = torch.cat([noise, cat, cont], 1).view(-1, self.latent_size, 1, 1)
        output = self.net(input)
        return output


class infogan_d_q_body_cats(nn.Module):
    def __init__(self, channels):
        super(infogan_d_q_body_cats, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
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
            nn.Conv2d(512, 1024, 4, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, input):
        output = self.net(input)
        return output


# Taken from: https://github.com/pianomania/infoGAN-pytorch/blob/master/trainer.py
# Used for uniform continuous code loss for InfoGAN
class log_gaussian:
    def __call__(self, mu, var, x):
        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
                (x-mu).pow(2).div(var.mul(2.0)+1e-6)
        return logli.sum(1).mean().mul(-1)
