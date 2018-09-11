import sys
sys.setrecursionlimit(2**25)
import keras.backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from __future__ import print_function
from six.moves import range
from keras.layers import Input, Dense,Dropout
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers.advanced_activations import PReLU
from keras.utils.generic_utils import Progbar
from sklearn.utils import shuffle
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.utils import np_utils

np.random.seed(31337)

# Hyper Parameters
FEATURES = 70
c_train_loss = []
c_train_accuracy = []
c_test_loss = []
c_test_accuracy = []


def build_classifier(dp=0.5, output_dim=39):
    cla = Sequential()
    cla.add(Dense(64, input_dim=FEATURES, kernel_initializer='glorot_uniform'))
    cla.add(PReLU())
    cla.add(Dropout(dp))

    cla.add(Dense(32, kernel_initializer='glorot_uniform'))
    cla.add(PReLU())
    cla.add(BatchNormalization())
    cla.add(Dropout(dp))

    cla.add(Dense(output_dim, kernel_initializer='glorot_uniform'))
    cla.add(Activation('softmax'))
    data_features = Input(shape=(FEATURES,))
    class_label = cla(data_features)

    return Model(input=data_features, outputs=class_label)


if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 1000
    batch_size = 128
    latent_size = 64
    train_test_ratio = 0.5
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the classifier
    classifier = build_classifier()
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    # get our crime data
    features_train = pd.read_csv("features_train.csv")
    labels_train = pd.read_csv("labels_train.csv")
    features_test = pd.read_csv("features_test.csv")
    labels_test = pd.read_csv("labels_test.csv")
    new_features_train = pd.read_csv("/data/generated_features.csv", skiprows=[0])
    new_labels_train = pd.read_csv("/data/generated_labels.csv", skiprows=[0])

    best_acc = 0.0
    best_loss = 100.00
    nb_epoch_for_best_acc = 0
    best_test_acc = 0.00
    total_epoch = 0

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(features_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_classifier_loss =[]

        for index in range(nb_batches):
            progress_bar.update(index)
            # get a batch of real data
            data_batch = features_train[index * batch_size:(index + 1) * batch_size]
            label_batch = labels_train[index * batch_size:(index + 1) * batch_size]
            new_data_batch = new_features_train[index * batch_size:(index + 1) * batch_size]
            new_label_batch = new_labels_train[index * batch_size:(index + 1) * batch_size]
            x = np.concatenate((data_batch, new_image_batch))
            y = np.concatenate((label_batch, new_label_batch))

            epoch_classifier_loss.append(classifier.train_on_batch(x, y))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        classifier_test_loss, classifier_test_acc = classifier.evaluate(features_test.values, labels_test.values,
                                                                        verbose=False)
        print("test loss for epoch{}:".format(epoch + 1), "classifier_test_loss = ", classifier_test_loss, "\n")
        print("test acc for epoch{}:".format(epoch + 1), "classifier_test_acc = ", classifier_test_acc, "\n")

        classifier_train_loss, classifier_train_acc = np.mean(np.array(epoch_classifier_loss), axis=0)
        print("train loss for epoch{}:".format(epoch + 1), "classifier_train_loss = ", classifier_train_loss, "\n")
        print("train acc for epoch{}:".format(epoch + 1), "classifier_train_acc = ", classifier_train_acc, "\n")
        c_train_loss.append(classifier_train_loss)
        c_test_loss.append(classifier_test_loss)
        c_train_accuracy.append(classifier_train_acc)
        c_test_accuracy.append(classifier_test_acc)

        if best_test_acc < classifier_test_acc:
            best_test_acc = classifier_test_acc
            nb_epoch_for_best_test_acc = epoch + 1

        print("the best test acc so far is : ", best_test_acc, " in epoch ", nb_epoch_for_best_test_acc)

        total_epoch = epoch + 1
        if classifier_train_loss < best_loss - 0.001:
            best_acc = classifier_train_acc
            best_loss = classifier_train_loss
            nb_epoch_for_best_acc = 0
        else:
            nb_epoch_for_best_acc += 1
            if nb_epoch_for_best_acc > 8:
                break

    print("best_loss:", best_loss)

    (loss, accuracy) = classifier.evaluate(features_test.values, labels_test.values, batch_size=batch_size, verbose=0)

    # show the accuracy on the testing set
    print("\n [INFO] Test accuracy: {:.2f}%".format(accuracy * 100))
    # plot the training process
    # x = np.arange(1, total_epoch + 1)
    #
    # ax1 = plt.subplot(211)
    # plt.plot(x, c_train_loss, c='red', label='training')
    # plt.plot(x, c_test_loss, c='blue', label='testing')
    # plt.ylabel('loss')
    # plt.legend(bbox_to_anchor=(0., 1.03, 1., .102), loc=3,
    #            ncol=2, mode="expand", borderaxespad=0.)
    # ax2 = plt.subplot(212)
    # plt.plot(x, c_train_accuracy, c='red', label='classifier_train_acc')
    # plt.plot(x, c_test_accuracy, c='blue', label='classifier_test_acc')
    # plt.ylabel('accuracy')
    # plt.xlabel('epochs')
    # plt.show()