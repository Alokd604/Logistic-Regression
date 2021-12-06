from sklearn import datasets
from random import shuffle, random, seed
import numpy as np
from math import log
import matplotlib.pyplot as plt


# Declaring necessary functions.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def shuffle_data(my_data):
    """"Takes a dataset as input.
    Divides and returns train_set, val_set and test_set"""
    train_set = []
    val_set = []
    test_set = []
    for s in my_data:
        r = random()
        if r >= 0 and r <= 0.7:
            train_set.append(s)
        elif r >= 0.7 and r <= 0.85:
            val_set.append(s)
        else:
            test_set.append(s)
    return train_set, val_set, test_set


def train(train_set, val_set, lr):  # lr = learing rate.
    X = []
    y = []
    for data in train_set:
        temp_x = list(data[0])
        temp_x.insert(0, 1)
        X.append(np.array(temp_x))
        y.append(data[1])
    theta = [random() for _ in range(len(X[0]))]  # random theta values.
    # Training
    train_loss = []
    graph_x = []
    for s in range(1000):
        TJ = 0
        for i in range(len(X)):
            Z = np.dot(X[i], theta)
            h = sigmoid(Z)
            J = -y[i] * log(h) - (1 - y[i]) * log(1 - h)
            TJ = TJ + J
            dv = X[i] * (h - y[i])
            theta = theta - dv * lr
            # theta = [(theta[i] - dv[i] * lr) for i in range(len(theta))]
        TJ = TJ / len(train_set)
        train_loss.append(TJ)
        graph_x.append(s)
    # plt.plot(graph_x, train_loss)
    # plt.show()

    # Validation.
    X = []
    y = []
    for data in val_set:
        temp_x = list(data[0])
        temp_x.insert(0, 1)
        X.append(np.array(temp_x))
        y.append(data[1])

    correct = 0
    for i in range(len(X)):
        Z = np.dot(X[i], theta)
        h = sigmoid(Z)
        if h >= 0.5:
            h = 1
        else:
            h = 0
        if h == y[i]:
            correct += 1
    val_acc = correct/len(val_set) * 100
    print(lr, val_acc)

# Main code starts here.
# loading dataset.
iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

# shuffling dataset together.
iris = list(zip(X, y))
shuffle(iris)

# splitting the data to 3 lists randomly.
train_set, val_set, test_set = shuffle_data(iris)
list_of_lr = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
for i in list_of_lr:
    train(train_set, val_set, i)


# train(train_set, test_set, 0.001)
