import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                          Activation, Embedding, multiply, concatenate,
                          MaxPooling2D, Reshape, Dense,
                          UpSampling2D, ZeroPadding2D,
                          BatchNormalization, LeakyReLU, Dropout,
                          Flatten, Lambda)
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import Progbar

from tensorflow import reduce_sum as tf_reduce_sum
from tensorflow import reduce_mean as tf_reduce_mean
from tensorflow import Print as tf_Print

from keras import backend as tf


# MNIST Constants:
IMG_ROWS = 28
IMG_COLS = 28
CHANNELS = 1
NUM_IMGS = 70000
C_CAT_SIZE = 10       # Size of discrete latent code
C_CONT_NUM = 2       # Number of continuous latent variables
NOISE_SIZE = 62
BATCH_SIZE = 128

def create_noise_grid():
    noise = np.zeros((10, 10, NOISE_SIZE))
    for i in range(10):
        noise[i, :] = np.random.normal(0, 1, NOISE_SIZE)
    return noise.reshape(100, NOISE_SIZE)

def create_label_grid():
    labs = np.zeros((10, 10)).astype(int)
    for i in range(C_CAT_SIZE):
        labs[i, :] = i
    return labs.reshape(100)

def create_continuous_grid():
    cont = np.zeros((10, 10, 2))
    for i in range(10):
        for j in range(10):
            cont[i, j] = [-2 + 0.4 * j, 0.0]
    return cont.reshape(100, 2)


NOISE_INPUTS = create_noise_grid()
C_CAT_FIXED = to_categorical(create_label_grid(), num_classes=C_CAT_SIZE)
C_CONT_FIXED = create_continuous_grid()



def build_g():
    # 3 inputs: noise, categorical var, continuous vars
    z = Input(shape=(NOISE_SIZE,))
    c_cat = Input(shape=(C_CAT_SIZE,))
    c_cont = Input(shape=(C_CONT_NUM,))

    # concatenate the inputs
    concat = concatenate([z, c_cat, c_cont], axis=-1)

    # build body of g
    model = Sequential([
        Dense(7 * 7 * 128, input_shape=(NOISE_SIZE + C_CAT_SIZE + C_CONT_NUM,)),
        Activation("relu"),
        Reshape((7, 7, 128)),
        BatchNormalization(),
        Conv2DTranspose(64, kernel_size=(4,4), strides=2, padding="same"),
        Activation("relu"),
        BatchNormalization(),
        Conv2DTranspose(1, kernel_size=(4,4), strides=2, padding="same"),
        Activation("tanh")
    ])

    output = model(concat)

    return Model(inputs=[z, c_cat, c_cont], outputs=[output])


def build_d_and_q():
    # 1 input: img
    img_shape = (IMG_ROWS, IMG_COLS, CHANNELS)
    img = Input(shape=img_shape)

    # build body of d and q (shared by d and q)
    body = Conv2D(64, kernel_size=(4,4), strides=2, padding="same", input_shape=img_shape, name="shared_conv1")(img)
    body = LeakyReLU(alpha=0.1, name="shared_lrelu1")(body)    # no batchnorm on first
    body = Dropout(0.25)(body)
    body = Conv2D(128, kernel_size=(4,4), strides=2, padding="same", name="shared_conv2")(body)
    body = LeakyReLU(alpha=0.1, name="shared_lrelu2")(body)
    body = Dropout(0.25)(body)
    body = BatchNormalization(name="shared_bnorm1")(body)
    body = Flatten(name="shared_flatten1")(body)


    # discriminator output 1 dense, sigmoid
    d_out = Dense(1, activation="sigmoid")(body)

    # Q net for prediction of latent codes
    q = Dense(128)(body)
    q = LeakyReLU(alpha=0.1)(q)
    q = BatchNormalization()(q)

    # categorical predictor (categorical crossentropy, so softmax)
    cat_out = Dense(C_CAT_SIZE, activation="softmax")(q)

    def exp_trick(x):
        return tf.exp(x)

    def exp_shape(input_shape):
        return input_shape

    # continuous predictor (need mean and stddev for all latent codes)
    #   first C_CONT_NUM nodes are mean, rest are std dev
    cont_out_mean = Dense(C_CONT_NUM, activation="linear")(q)
    cont_out_var = Dense(C_CONT_NUM, activation="linear")(q)
    cont_out_var = Lambda(exp_trick, output_shape=exp_shape)(cont_out_var)
    cont_out = concatenate([cont_out_mean, cont_out_var])


    # Output both the discriminator model and the q-net model
    return Model(inputs=[img], outputs=[d_out]), Model(inputs=[img], outputs=[cat_out, cont_out])


# OpenAI https://github.com/openai/InfoGAN/blob/master/infogan/misc/distributions.py
#  Loss function for gaussian negative log likelihood
#  Paper says to use this even for uniform continuous latent variable
def logli(y_true, y_pred):
    # Assuming y_pred is (n_batches, C_CONT_NUM * 2) but could be very wrong about that
    mean = y_pred[:, 0:2]
    stddev = y_pred[:, 2:]
    epsilon = (y_true - mean) / (stddev + tf.epsilon())
    loss_out = tf_reduce_sum(-0.5 * np.log(2*np.pi) - tf.log(stddev + tf.epsilon()) - 0.5*tf.square(epsilon),
                             reduction_indices=1,)
    return loss_out



def gaussian_loss(y_true, y_pred):

    mean = y_pred[:, 0:2]
    stddev = y_pred[:, 2:]
    epsilon = (y_true - mean) / (tf.exp(stddev) + tf.epsilon())
    loss_Q_C = (stddev + 0.5 * tf.square(epsilon))
    loss_Q_C = tf.mean(loss_Q_C)

    return loss_Q_C

def loss_gaussian(y_true, y_pred):
    # Assuming y_pred is (n_batches, C_CONT_NUM * 2) but could be very wrong about that
    mean = y_pred[:, 0:2]
    var = y_pred[:, 2:]

    epsilon = y_true - mean

    logli = -0.5*tf.log(2*np.pi*var + tf.epsilon()) - (tf.square(epsilon) / (2 * var + tf.epsilon()))
    logli = tf_reduce_sum(logli, 1)
    logli = -1 * tf_reduce_mean(logli)

    return logli



# initialize models
def setup_models():
    # optimizer parameters specified in paper
    G_opt = Adam(1e-3, 0.5)
    D_opt = Adam(2e-4, 0.5)

    # Build models
    G = build_g()
    D, Q = build_d_and_q()

    # Compile classifier models
    G.compile(loss="binary_crossentropy", optimizer=G_opt)
    D.compile(loss="binary_crossentropy", optimizer=D_opt, metrics=["accuracy"])


    # Make combined model for g and q training
    D.trainable = False
    for layer in D.layers:
        layer.trainable=False
    z = Input(shape=(NOISE_SIZE,))
    c_cat = Input(shape=(C_CAT_SIZE,))
    c_cont = Input(shape=(C_CONT_NUM,))
    G_out = G([z, c_cat, c_cont])
    real_or_fake = D(G_out)
    c_cat_out, c_cont_out = Q(G_out)

    combined = Model(inputs=[z, c_cat, c_cont], outputs=[real_or_fake, c_cat_out, c_cont_out])

    combined.compile(loss=["binary_crossentropy", "categorical_crossentropy", loss_gaussian],
                     optimizer=G_opt, loss_weights=[1,1,0.01])

    return G, D , combined


def save_images(generator, epoch):
    gen_imgs = generator.predict([NOISE_INPUTS, C_CAT_FIXED, C_CONT_FIXED])
    gen_imgs = 0.5 * gen_imgs + 0.5
    gen_imgs = gen_imgs.reshape((-1, IMG_ROWS, IMG_COLS))


    plt.figure(figsize=(10, 10))
    for i in range(100):
        # display reconstruction

        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(gen_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("/home/ubuntu/photos/infogan-mnist/infogan_mnist_%d.png" % epoch)
    plt.close()


def train(G, D, combined, epochs=50):

    # Load and normalize data:
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = np.concatenate((x_train, x_test), axis=0)
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train.reshape((-1, IMG_ROWS, IMG_COLS, CHANNELS))

    # Number of batch loops:
    batch_loops = int(NUM_IMGS // BATCH_SIZE)

    # Train InfoGAN:
    for epoch in range(epochs):
        shuffle_idx = np.random.permutation(NUM_IMGS)
        real_imgs = x_train[shuffle_idx]

        progress_bar = Progbar(target=batch_loops)

        for batch_i in range(batch_loops):
            progress_bar.update(batch_i)

            # Discriminator:
            real_img_batch = real_imgs[batch_i*BATCH_SIZE:(batch_i+1)*BATCH_SIZE]
            noise_batch = np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
            c_cat_batch = np.random.randint(0, C_CAT_SIZE, BATCH_SIZE)
            c_cat_batch = to_categorical(c_cat_batch, num_classes=C_CAT_SIZE)
            c_cont_batch = np.random.uniform(-1, 1, (BATCH_SIZE, C_CONT_NUM))
            fake_img_batch = G.predict([noise_batch, c_cat_batch, c_cont_batch])

            d_loss_real = D.train_on_batch(real_img_batch, np.ones(BATCH_SIZE))
            d_loss_fake = D.train_on_batch(fake_img_batch, np.zeros(BATCH_SIZE))
            d_loss_total = np.add(d_loss_real, d_loss_fake) * 0.5

            # Generator:
            noise_batch = np.random.normal(0, 1, (2*BATCH_SIZE, NOISE_SIZE))
            c_cat_batch = np.random.randint(0, C_CAT_SIZE, 2*BATCH_SIZE)
            c_cat_batch = to_categorical(c_cat_batch, num_classes=C_CAT_SIZE)
            c_cont_batch = np.random.uniform(-1, 1, (2*BATCH_SIZE, C_CONT_NUM))
            g_loss = combined.train_on_batch([noise_batch, c_cat_batch, c_cont_batch],
                                             [np.ones(2*BATCH_SIZE), c_cat_batch, c_cont_batch])
#             print(d_loss_total)
#             print(g_loss)

        print(epoch)
        print(d_loss_total)
        print(g_loss)
        save_images(G, epoch)



G, D, combined = setup_models()

g = train(G, D, combined, 100)
