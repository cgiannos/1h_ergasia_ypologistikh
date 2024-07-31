import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from keras.utils import np_utils
from matplotlib import pyplot as plt, pyplot
from numpy import mean, std
from numpy.polynomial.tests.test_classes import random
from sklearn.model_selection import KFold
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from tensorflow.python.keras.optimizer_v1 import sgd
from tensorflow.python.keras.regularizers import l2

m_train = pd.read_csv('mnist_train.csv')
m_test = pd.read_csv('mnist_test.csv')

x_train = m_train.drop('label', axis=1)
y_train = m_train['label']
x_test = m_test.drop('label', axis=1)
y_test = m_test['label']

# reshape dataset to have a single channel
x_train = x_train.values.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.values.reshape((x_test.shape[0], 28, 28, 1))

image_size = 784  # 28*28
n = 10

# normalization

x_test = x_test / 255
x_train = x_train / 255

# one-hot encoding

y_train = np_utils.to_categorical(y_train, n)
y_test = np_utils.to_categorical(y_test, n)


# NN
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.6)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])  # loss='categorigal_crossentropy' or loss='mse'
    return model


# kfold
#kfold = KFold(5, shuffle=True, random_state=1)
accuracies, histories, losses = list(), list(), list()


def train(models):
    for i in range(len(models)):
        model = define_model()
       # trainX, trainY, testX, testY = x_train[train_ix], y_train[train_ix], x_train[test_ix], y_train[test_ix]

        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        history = models[i].fit(x=x_train, batch_size=32, epochs=1, verbose=0, validation_data=(x_test, y_test))
        loss, accuracy = model.evaluate(x_test,y_test, verbose=0)
        # print('> %.3f' % (accuracy * 100.0))
        accuracies.append(accuracy)
        histories.append(history)
        losses.append(loss)
        return model, losses


no_of_generations = 10
no_of_individuals = 10
mutate_factor = 0.05
individuals = []

layers = [0, 3, 5]


def mutate(new_individual):
    for i in layers:
        for bias in range(len(new_individual.layers[i].get_weights()[1])):
            r = random.random()
            if (r < mutate_factor):
                new_individual.layers[i].get_weights()[1][bias] *= random.uniform(-0.5, 0.5)

    for i in layers:
        for weight in new_individual.layers[i].get_weights()[0]:
            r = random.random()
            if (r < mutate_factor):
                for j in range(len(weight)):
                    if (random.random() < mutate_factor):
                        new_individual.layers[i].get_weights()[0][j] *= random.uniform(-0.5, 0.5)

    return new_individual


def crossover(individuals):
    new_individuals = []

    new_individuals.append(individuals[0])
    new_individuals.append(individuals[1])

    for i in range(2, no_of_individuals):
        if (i < (no_of_individuals - 2)):
            if (i == 2):
                parentA = random.choice(individuals[:3])
                parentB = random.choice(individuals[:3])
            else:
                parentA = random.choice(individuals[:])
                parentB = random.choice(individuals[:])

            for i in layers:
                temp = parentA.layers[i].get_weights()[1]
                parentA.layers[i].get_weights()[1] = parentB.layers[i].get_weights()[1]
                parentB.layers[i].get_weights()[1] = temp

                new_individual = random.choice([parentA, parentB])

        else:
            new_individual = random.choice(individuals[:])

        new_individuals.append(mutate(new_individual))
        # new_individuals.append(new_individual)

    return new_individuals


def evolve(individuals, losses):
    sorted_y_idx_list = sorted(range(len(losses)), key=lambda x: losses[x])
    individuals = [individuals[i] for i in sorted_y_idx_list]

    # winners = individuals[:6]

    new_individuals = crossover(individuals)

    return new_individuals


for i in range(no_of_individuals):
    individuals.append(define_model())

for generation in range(no_of_generations):
    individuals, losses = train(individuals)
    print(losses)

    individuals = evolve(individuals, losses)

# mean
#print('losses: mean=%.3f std=%.3f, n=%d' % (mean(losses) * 100, std(losses) * 100, len(losses)))

# plot
for i in range(len(histories)):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(histories[i].history['loss'], color='blue', label='train')
    pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.title('Classification Accuracy')
    pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
    pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
pyplot.show()
