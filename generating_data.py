import cPickle as pickle
import sys
import keras.backend as K
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from __future__ import print_function
from collections import defaultdict
from sklearn.cross_validation import StratifiedShuffleSplit
from copy import deepcopy
from datetime import datetime
from six.moves import range
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.initializers import RandomNormal
from sklearn import preprocessing
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.utils import np_utils
sys.setrecursionlimit(2**25)
np.random.seed(31337)

# Hyper Parameters
FEATURES = 70
c_train_loss = []
c_test_loss = []
g_train_loss = []
d_train_true_loss = []
d_train_fake_loss = []


def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)


def build_generator(latent_size):
    # weights are initlaized from normal distribution with below params
    weight_init = RandomNormal(mean=0., stddev=0.02)
    gen = Sequential()

    gen.add(Dense(1024, input_dim=latent_size))
    gen.add(LeakyReLU())

    gen.add(Dense(128))
    gen.add(LeakyReLU())

    gen.add(Dense(FEATURES, activation='tanh', kernel_initializer=weight_init))

    latent = Input(shape=(latent_size, ))

    # this will be our label (0~38)
    data_class = Input(shape=(1,), dtype='int32')

    # 39 classes in SF Crime
    cls = Flatten()(Embedding(39, latent_size, init='glorot_normal')(data_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_data = gen(h)

    return Model(input=[latent, data_class], output=fake_data, name='G')


def build_discriminator():
    # weights are initlaized from normal distribution with below params
    weight_init = RandomNormal(mean=0., stddev=0.02)

    dis = Sequential()

    dis.add(Dense(128, input_dim=FEATURES, kernel_initializer=weight_init))
    dis.add(LeakyReLU())
    dis.add(Dropout(0.3))

    dis.add(Dense(32, kernel_initializer=weight_init))
    dis.add(LeakyReLU())
    dis.add(Dropout(0.3))

    dis.add(Dense(1, activation='linear'))

    data_features = Input(shape=(FEATURES, ))

    is_fake = dis(data_features)

    return Model(input=data_features, output=is_fake, name='D')


if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 30
    d_iters = 6
    batch_size = 128
    latent_size = 100
    train_test_ratio = 0.5
    fake_ratio = 1
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    discriminator = build_discriminator()
    opt = RMSprop(lr=0.00005)
    discriminator.compile(
        optimizer=opt,
        loss=wasserstein)

    latent = Input(shape=(latent_size,), name='input_z_')
    data_class = Input(shape=(1,), dtype='int32', name='input_class_')

    # build the generator
    generator = build_generator(latent_size)

    # get a fake tuple
    fake_tuple = generator([latent, data_class])
    fake = discriminator(fake_tuple)

    combined = Model(input=[latent, data_class], output=[fake])
    combined.get_layer('D').trainable = False

    combined.compile(
        optimizer=RMSprop(lr=0.00005),
        loss=[wasserstein]
    )

    # get our crime data
    features = pd.read_csv("features.csv")
    labels = pd.read_csv("labels.csv")
    train_size = int(train_test_ratio * len(features))
    features_train = features[0:train_size]
    labels_train = labels[0:train_size]
    features_test = features[train_size:len(features)]
    labels_test = labels[train_size:len(features)]

    nb = int(features_train.shape[0] / batch_size) * nb_epochs
    rounds = int(features_train.shape[0] / batch_size)

    best_acc = 0.0
    best_loss = 100.00
    nb_epoch_for_best_acc = 0
    best_test_acc = 0.00
    total_epoch = 0
    epoch_gen_loss = []
    epoch_disc_true_loss = []
    epoch_disc_fake_loss = []

    progress_bar = Progbar(target=nb)

    for index in range(nb):
        progress_bar.update(index)
        for d_it in range(d_iters):
            # unfreeze D
            discriminator.trainable = True
            for l in discriminator.layers: l.trainable = True

            # clip D weights
            for l in discriminator.layers:
                weights = l.get_weights()
                weights = [np.clip(w, -0.01, 0.01) for w in weights]
                l.set_weights(weights)

            # 1.1: maximize D output on reals === minimize -1*(D(real))
            # get a batch of real data
            data_index = np.random.choice(len(features_train), batch_size, replace=False)
            data_batch = features_train.values[data_index]
            label_batch = labels_train.values[data_index]

            epoch_disc_true_loss.append(discriminator.train_on_batch(data_batch, -np.ones(batch_size)))

            # 1.2: minimize D output on fakes

            # generate a new batch of noise
            noise = np.random.normal(loc=0.0, scale=1, size=(int(batch_size * fake_ratio), latent_size))
            # sample some labels from p_c
            sampled_labels = np.random.randint(0, 39, int(batch_size * fake_ratio))
            aux_sampled_labels = np_utils.to_categorical(sampled_labels, 39)
            generated_data = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)
            epoch_disc_fake_loss.append(
                discriminator.train_on_batch(generated_data, np.ones(int(batch_size * fake_ratio))))

        # freeze D and C
        discriminator.trainable = False
        for l in discriminator.layers: l.trainable = False

        # generate a new batch of noise
        noise = np.random.normal(loc=0.0, scale=1, size=(int(batch_size * fake_ratio), latent_size))
        # sample some labels from p_c
        sampled_labels = np.random.randint(0, 39, int(batch_size * fake_ratio))
        aux_sampled_labels = np_utils.to_categorical(sampled_labels, 39)

        epoch_gen_loss.append(combined.train_on_batch(
            [noise, sampled_labels], [-np.ones(int(batch_size * fake_ratio))]))

    result_features = np.arange(FEATURES)
    result_labels = np.arange(39)

    for index in range(rounds):
        noise = np.random.normal(loc=0.0, scale=1, size=(int(batch_size * fake_ratio), latent_size))
        # sample some labels from p_c
        sampled_labels = np.random.randint(0, 39, int(batch_size * fake_ratio))
        aux_sampled_labels = np_utils.to_categorical(sampled_labels, 39)
        generated_data = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=0)
        result_features = np.vstack((result_features, generated_data))
        result_labels = np.vstack((result_labels, aux_sampled_labels))

    with open('/data/generated_features.csv', 'w') as wf:
        for line in result_features:
            print(",".join([str(x) for x in line]), file=wf)

    with open('/data/generated_labels.csv', 'w') as wf:
        for line in result_labels:
            print(",".join([str(x) for x in line]), file=wf)


