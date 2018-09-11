import time
from sklearn import metrics
import pickle as pickle
import pandas as pd
import numpy as np
from keras.utils import np_utils

train_test_ratio = 0.5


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


def read_data():
    features_train = pd.read_csv("features_train.csv")
    labels_train = pd.read_csv("labels_train.csv")
    features_test = pd.read_csv("features_test.csv")
    labels_test = pd.read_csv("labels_test.csv")
    new_features_train = pd.read_csv("/data/generated_features.csv", skiprows=[0])
    new_labels_train = pd.read_csv("/data/generated_labels.csv", skiprows=[0])
    x = np.concatenate((features_train, new_features_train))
    y = np.concatenate((labels_train, new_labels_train))
    return x, y, features_test.values, labels_test.values


if __name__ == '__main__':

    test_classifiers = ['RF', 'DT']
    classifiers = {
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier
                   }

    print('reading training and testing data...')
    train_x, train_y, test_x, test_y = read_data()
    train_y = np.argmax(train_y, axis=1)
    test_y = np.argmax(test_y, axis=1)

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))
