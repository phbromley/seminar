import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Keras Imports
from keras.models import Sequential, Model
from keras.layers import (Input, Conv2D, Activation,
                          MaxPooling2D, Reshape, Dense,
                          UpSampling2D, ZeroPadding2D,
                          BatchNormalization, LeakyReLU, Dropout,
                          Flatten)
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar

from keras.datasets import mnist
from keras.datasets import cifar10



# Much inspiration taken from: https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

# Constants:
IMG_ROWS = 28
IMG_COLS = 28
CHANNELS = 1
NUM_IMGS = 70000
LATENT_SIZE = 100
BATCH_SIZE = 128
NOISE_INPUTS = np.random.normal(0, 1, (100, LATENT_SIZE))  # for saving images

def build_discriminator():
    input_shape = (IMG_ROWS, IMG_COLS, CHANNELS)

    model = Sequential([

        Conv2D(32, kernel_size=3, strides=2, input_shape=input_shape, padding="same"),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(64, kernel_size=3, strides=2, padding="same"),
        ZeroPadding2D(padding=((0,1),(0,1))),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        BatchNormalization(momentum=0.8),
        Conv2D(128, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        BatchNormalization(momentum=0.8),
        Conv2D(256, kernel_size=3, strides=1, padding="same"),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    img = Input(shape=input_shape)
    discriminator = model(img)

    return Model(img, discriminator)


def build_generator():
    z_shape = (LATENT_SIZE,)

    model = Sequential([
        Dense(128 * 7 * 7, activation="relu", input_shape=z_shape),
        Reshape((7, 7, 128)),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(128, kernel_size=3, padding="same"),
        Activation("relu"),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(64, kernel_size=3, padding="same"),
        Activation("relu"),
        BatchNormalization(),
        Conv2D(1, kernel_size=3, padding="same"),
        Activation("tanh")
    ])

    z = Input(shape=z_shape)
    img = model(z)

    return Model(z, img)

def setup_models():
    data_shape = (IMG_ROWS, IMG_COLS, CHANNELS)

    optimizer = Adam(0.0002, 0.5)
    # Build discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    # Build and compile the generator
    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    z = Input(shape=(LATENT_SIZE,))
    synthetic_img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    valid = discriminator(synthetic_img)

    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    return generator, discriminator, combined


def save_images(generator, epoch):
    gen_imgs = generator.predict(NOISE_INPUTS)
    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = gen_imgs.reshape((-1, 28, 28))


    plt.figure(figsize=(10, 10))
    for i in range(100):
        # display reconstruction

        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(gen_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("/home/ubuntu/photos/dcgan-mnist/dcgan_mnist_%d.png" % epoch)
    plt.close()




def train(generator, discriminator, combined, epochs=50):
    # Load and normalize data:
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = np.concatenate((x_train, x_test), axis=0)
    x_train = x_train.reshape((NUM_IMGS, IMG_ROWS, IMG_COLS, CHANNELS))
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    # Label arrays for positive/negative examples
    positive_examples = np.ones((BATCH_SIZE, 1))
    negative_examples = np.zeros((BATCH_SIZE, 1))

    # Number of batch loops:
    batch_loops = int(NUM_IMGS // BATCH_SIZE)

    for epoch in range(epochs):
        progress_bar = Progbar(target=batch_loops)

        shuffle_idx = np.random.permutation(NUM_IMGS)
        real_imgs = x_train[shuffle_idx]

        for batch_i in range(batch_loops):
            progress_bar.update(batch_i)

            # Discriminator:
            img_batch = real_imgs[batch_i*BATCH_SIZE:(batch_i+1)*BATCH_SIZE]
            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_SIZE))
            fake_img_batch = generator.predict(noise)


            d_loss_real = discriminator.train_on_batch(img_batch, positive_examples)
            d_loss_fake = discriminator.train_on_batch(fake_img_batch, negative_examples)
            d_loss_total = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Generator:
            noise = np.random.normal(0, 1, (2*BATCH_SIZE, LATENT_SIZE))
            positive = np.concatenate((positive_examples, positive_examples), axis=0)

            g_loss = combined.train_on_batch(noise, positive)

        print ("Epoch: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss_total[0], 100*d_loss_total[1], g_loss))
        if epoch % 2 == 0:
            save_images(generator, epoch)

    return generator


# Example training run
# generator, discriminator, combined = setup_models()
# g = train(generator, discriminator, combined, 50)
