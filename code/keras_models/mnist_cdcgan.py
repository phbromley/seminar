import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                          Activation, Embedding, multiply, concatenate,
                          MaxPooling2D, Reshape, Dense,
                          UpSampling2D, ZeroPadding2D,
                          BatchNormalization, LeakyReLU, Dropout,
                          Flatten)
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar

from keras.datasets import mnist
from keras.datasets import cifar10

from keras import backend as tf


# Constants:
IMG_ROWS = 28
IMG_COLS = 28
CHANNELS = 1
NUM_IMGS = 70000
NUM_CLASSES = 10
LATENT_SIZE = 100
BATCH_SIZE = 120

def create_label_grid():
    labs = np.zeros((10, 10)).astype(int)
    for i in range(10):
        labs[i, :] = i
    return labs.reshape(100, 1)

NOISE_INPUTS = np.random.normal(0, 1, (100, LATENT_SIZE))  # for saving images
LABEL_INPUTS = create_label_grid()

# Generator:
def build_generator():

    # 2 inputs: noise and label
    z = Input(shape=(LATENT_SIZE,))
    y = Input(shape=(1,))

    # Make embedding layers and make joint hidden representation
    embedding = Embedding(NUM_CLASSES, LATENT_SIZE, input_length=1)(y)
    embedding = Flatten()(embedding)
    joint_hidden = multiply([z, embedding])

    # Body of generator
    model = Sequential([
        Dense(128 * 7 * 7, activation="relu", input_shape=(LATENT_SIZE,)),
        Reshape((7, 7, 128)),
        BatchNormalization(momentum=0.8),
        UpSampling2D(),
        Conv2D(128, kernel_size=3, padding="same"),
        Activation("relu"),
        BatchNormalization(momentum=0.8),
        UpSampling2D(),
        Conv2D(64, kernel_size=3, padding="same"),
        Activation("relu"),
        BatchNormalization(momentum=0.8),
        Conv2D(1, kernel_size=3, padding="same"),
        Activation("tanh")
    ])
    model.summary()

    output = model(joint_hidden)

    generator = Model([z, y], output)

    return generator


# Discriminator
def build_discriminator():

    # 2 inputs: image and label
    img = Input(shape=(IMG_ROWS, IMG_COLS, CHANNELS))
    y = Input(shape=(1,))

    # Make embedding layers, upsample to add as fourth channel
    embedding = Embedding(NUM_CLASSES, 49, input_length=1)(y)
    embedding = Flatten()(embedding)
    embedding = Reshape((7, 7, 1))(embedding)
    embedding = UpSampling2D(size=(4,4))(embedding)
    joint_hidden = concatenate([img, embedding], axis=-1)

    # Body of discriminator
    model = Sequential([
        Conv2D(32, (3,3), strides=2, input_shape=(IMG_ROWS, IMG_COLS, CHANNELS+1), padding="same"),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Conv2D(64, (3,3), strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        BatchNormalization(momentum=0.8),
        Conv2D(128, (3,3), strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        BatchNormalization(momentum=0.8),
        Conv2D(256, (3,3), strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Dropout(0.25),
        Flatten(),
        Dense(1, activation="sigmoid")
    ])

    model.summary()

    pred = model(joint_hidden)


    discriminator = Model([img, y], pred)
    discriminator.summary()
    return discriminator


def setup_models():

    opt = optimizer = Adam(0.0002, 0.5)     # parameters specified in DCGAN paper

    # Build models
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile models
    generator.compile(loss="binary_crossentropy", optimizer=opt)
    discriminator.compile(loss="binary_crossentropy",
                          optimizer=opt,
                          metrics=["accuracy"])

    # Build combined model for generator training (freeze discriminator weights)
    discriminator.trainable = False

    z = Input(shape=(LATENT_SIZE,))
    y = Input(shape=(1,))
    noise_to_img = generator([z, y])
    img_to_pred = discriminator([noise_to_img, y])
    combined_model = Model([z, y], img_to_pred)

    # Compile combined model for generator training
    combined_model.compile(loss="binary_crossentropy", optimizer=opt)

    return generator, discriminator, combined_model



def save_images(generator, epoch):
    gen_imgs = generator.predict([NOISE_INPUTS, LABEL_INPUTS])
    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = gen_imgs.reshape((-1, 28, 28))


    plt.figure(figsize=(10, 10))
    for i in range(100):
        # display reconstruction

        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(gen_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("/home/ubuntu/photos/cdcgan-mnist/cdcgan_mnist_%d.png" % epoch)
    plt.close()



def train(generator, discriminator, combined, epochs):

    # Load and normalize data:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.concatenate((x_train, x_test), axis=0)
    x_train = x_train.reshape((NUM_IMGS, IMG_ROWS, IMG_COLS, CHANNELS))
    y_train = np.concatenate((y_train, y_test), axis=0)
    y_train = y_train.reshape((-1, 1))
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    # Label arrays for positive/negative examples
    positive_examples = np.ones((BATCH_SIZE, 1))
    negative_examples = np.zeros((BATCH_SIZE, 1))

    # Number of batch loops:
    batch_loops = int(NUM_IMGS // BATCH_SIZE)

    # Train cGAN:
    for epoch in range(epochs):
        progress_bar = Progbar(target=batch_loops)

        shuffle_idx = np.random.permutation(NUM_IMGS)
        real_imgs = x_train[shuffle_idx]
        labels = y_train[shuffle_idx]

        for batch_i in range(batch_loops):
            progress_bar.update(batch_i)

            # Discriminator:
            img_batch = real_imgs[batch_i*BATCH_SIZE:(batch_i+1)*BATCH_SIZE]
            label_batch = labels[batch_i*BATCH_SIZE:(batch_i+1)*BATCH_SIZE]
            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_SIZE))
            fake_img_batch = generator.predict([noise, label_batch])


            d_loss_real = discriminator.train_on_batch([img_batch, label_batch],
                                                        positive_examples)
            d_loss_fake = discriminator.train_on_batch([fake_img_batch, label_batch],
                                                        negative_examples)
            d_loss_total = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Generator:
            noise = np.random.normal(0, 1, (2*BATCH_SIZE, LATENT_SIZE))
            fake_labels = np.random.randint(0, NUM_CLASSES, (2*BATCH_SIZE, 1))
            positive = np.concatenate((positive_examples, positive_examples), axis=0)

            g_loss = combined.train_on_batch([noise, fake_labels], positive)


        print ("Epoch: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss_total[0], 100*d_loss_total[1], g_loss))
        save_images(generator, epoch)

    return generator


# For training: 
# generator, discriminator, combined = setup_models()
#
# g = train(generator, discriminator, combined, 50)
