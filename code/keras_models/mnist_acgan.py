import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                          Activation, Embedding, multiply, concatenate,
                          MaxPooling2D, Reshape, Dense,
                          UpSampling2D, ZeroPadding2D,
                          BatchNormalization, LeakyReLU, Dropout,
                          Flatten, GaussianNoise)
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
LATENT_SIZE = 110
BATCH_SIZE = 100

def create_label_grid():
    labs = np.zeros((10, 10)).astype(int)
    for i in range(10):
        labs[i, :] = i
    return labs.reshape(100, 1)

NOISE_INPUTS = np.random.normal(0, 1, (100, LATENT_SIZE))  # for saving images
LABEL_INPUTS = create_label_grid()




# Generator:
def build_generator():
    '''G(z, y) -> synthetic image'''

    # 2 inputs: noise and label
    z = Input(shape=(LATENT_SIZE,))
    y = Input(shape=(1,))

    # Make embedding layers and make joint hidden representation
    embedding = Embedding(NUM_CLASSES, LATENT_SIZE, input_length=1,
                          embeddings_initializer='glorot_normal')(y)
    embedding = Flatten()(embedding)
    joint_hidden = multiply([z, embedding])

    # Body of generator
    model = Sequential([
        Dense(192 * 7 * 7, input_dim=LATENT_SIZE, activation='relu',
                           kernel_initializer='glorot_normal'),
        Reshape((7, 7, 192)),
        UpSampling2D(),
        Conv2D(96, kernel_size=5, padding='same', activation='relu',
                    kernel_initializer='glorot_normal'),
        UpSampling2D(),
        Conv2D(1, kernel_size=5, padding='same', activation='tanh',
                  kernel_initializer='glorot_normal'),
    ])
    model.summary()

    output = model(joint_hidden)

    generator = Model([z, y], output)

    return generator


# Discriminator
def build_discriminator():
    '''D(img) -> p1, p2 where p1 = fake/real prediction, p2 = auxiliary label prediction'''

    # Input: image
    img = Input(shape=(IMG_ROWS, IMG_COLS, CHANNELS))

    # Body of discriminator
    model = Sequential([
        GaussianNoise(0.05, input_shape=(IMG_ROWS, IMG_COLS, CHANNELS)),
        Conv2D(16, (3,3), strides=2, input_shape=(IMG_ROWS, IMG_COLS, CHANNELS), padding="same",
                   kernel_initializer='glorot_normal'),
        LeakyReLU(alpha=0.2),
        Dropout(0.5),
        Conv2D(32, (3,3), strides=1, padding="same", kernel_initializer='glorot_normal'),
        LeakyReLU(alpha=0.2),
        Dropout(0.5),
        BatchNormalization(momentum=0.8),
        Conv2D(64, (3,3), strides=2, padding="same", kernel_initializer='glorot_normal'),
        LeakyReLU(alpha=0.2),
        Dropout(0.5),
        BatchNormalization(momentum=0.8),
        Conv2D(128, (3,3), strides=1, padding="same", kernel_initializer='glorot_normal'),
        LeakyReLU(alpha=0.2),
        Dropout(0.5),
        BatchNormalization(momentum=0.8),
        Conv2D(256, (3,3), strides=2, padding="same", kernel_initializer='glorot_normal'),
        LeakyReLU(alpha=0.2),
        Dropout(0.5),
        BatchNormalization(momentum=0.8),
        Conv2D(512, (3,3), strides=1, padding="same", kernel_initializer='glorot_normal'),
        LeakyReLU(alpha=0.2),
        Dropout(0.5),
        BatchNormalization(momentum=0.8),
        Flatten()
    ])

    # attach input to model
    full_model = model(img)

    # Two outputs: real/fake binary prediction, and label multiclass prediction
    real_or_fake = Dense(1, activation="sigmoid",
                            kernel_initializer='glorot_normal')(full_model)        # sigmoid used for binary logistic reg.
    label = Dense(NUM_CLASSES, activation="softmax",
                               kernel_initializer='glorot_normal')(full_model)     # softmax used for multiclass logistic reg.

    discriminator = Model(img, [real_or_fake, label])
    return discriminator


def setup_models():

    opt = Adam(0.0002, 0.5)     # parameters specified in DCGAN paper

    # Build models
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile models
    generator.compile(loss="binary_crossentropy", optimizer=opt)
    discriminator.compile(loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
                          optimizer=opt,
                          metrics=["accuracy"])

    # Build combined model for generator training (freeze discriminator weights)
    discriminator.trainable = False

    z = Input(shape=(LATENT_SIZE,))
    y = Input(shape=(1,), dtype="int32")
    noise_to_img = generator([z, y])
    real_or_fake, label = discriminator(noise_to_img)
    combined_model = Model([z, y], [real_or_fake, label])

    # Compile combined model for generator training
    combined_model.compile(loss=["binary_crossentropy", "sparse_categorical_crossentropy"], optimizer=opt)

    return generator, discriminator, combined_model



def save_images(generator, epoch):
    gen_imgs = generator.predict([NOISE_INPUTS, LABEL_INPUTS])
    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = gen_imgs.reshape((-1, IMG_ROWS, IMG_COLS))


    plt.figure(figsize=(10, 10))
    for i in range(100):
        # display reconstruction

        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(gen_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("/home/ubuntu/photos/acgan-mnist/acgan_mnist_%d.png" % epoch)
    plt.close()



def train(generator, discriminator, combined, epochs=50):

    # Load and normalize data:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.concatenate((x_train, x_test), axis=0)
    x_train = x_train.reshape((NUM_IMGS, IMG_ROWS, IMG_COLS, CHANNELS))
    y_train = np.concatenate((y_train, y_test), axis=0)
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5

    # Number of batch loops:
    batch_loops = int(NUM_IMGS // BATCH_SIZE)

    # Train ACGAN:
    for epoch in range(epochs):
        shuffle_idx = np.random.permutation(NUM_IMGS)
        real_imgs = x_train[shuffle_idx]
        labels = y_train[shuffle_idx]

        progress_bar = Progbar(target=batch_loops)

        for batch_i in range(batch_loops):
            progress_bar.update(batch_i)

            pos_examples_smooth = np.random.normal(0.7, 0.12, (BATCH_SIZE,))
            neg_examples_smooth = np.random.normal(0.0, 0.3, (BATCH_SIZE,))

            # Discriminator:
            real_img_batch = real_imgs[batch_i*BATCH_SIZE:(batch_i+1)*BATCH_SIZE]
            real_label_batch = labels[batch_i*BATCH_SIZE:(batch_i+1)*BATCH_SIZE].reshape(-1,)
            noise_batch = np.random.normal(0, 1, (BATCH_SIZE, LATENT_SIZE))
            fake_label_batch = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
            fake_img_batch = generator.predict([noise_batch, fake_label_batch.reshape((-1, 1))])

            d_loss_r = discriminator.train_on_batch(real_img_batch, [pos_examples_smooth, real_label_batch])
            d_loss_f = discriminator.train_on_batch(fake_img_batch, [neg_examples_smooth, fake_label_batch])
            d_loss_total = np.add(d_loss_r, d_loss_f) * 0.5

            # Generator:
            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_SIZE))
            fake_labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE)
            pos_examples_smooth = np.random.normal(0.7, 0.12, (BATCH_SIZE,))
            g_loss = combined.train_on_batch([noise, fake_labels.reshape(-1, 1)], [pos_examples_smooth, fake_labels])

#         print ("Epoch: %d [D f/r loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
        print(d_loss_total)
        print(g_loss)
        save_images(generator, epoch)


# For training:
# generator, discriminator, combined = setup_models()
#
# g = train(generator, discriminator, combined, 50)
