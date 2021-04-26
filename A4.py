import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from keras.utils import np_utils
from matplotlib import pyplot as plt, pyplot
from numpy import mean, std
from sklearn.model_selection import KFold
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizer_v1 import sgd
from tensorflow.python.keras.regularizers import l2

m_train = pd.read_csv('mnist_train.csv')
m_test = pd.read_csv('mnist_test.csv')

x_train = m_train.drop('label', axis=1)
y_train = m_train['label']
x_test = m_test.drop('label', axis=1)
y_test = m_test['label']

image_size = 784  # 28*28
n = 10

# normalization

x_test = x_test / 255
x_train = x_train / 255

# one-hot encoding

y_train = np_utils.to_categorical(y_train, n)
y_test = np_utils.to_categorical(y_test, n)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()


# NN
def define_model():
    model = Sequential()

   
    model.add(Dense(units=794, activation='sigmoid', kernel_regularizer=l2(0.1), input_shape=(image_size,))) 
   # model.add(Dense(units=10, activation='sigmoid'))
    model.add(Dense(units=n, activation='softmax'))
    model.summary()

    opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])      #loss='categorigal_crossentropy' or loss='mse'
    return model

#kfold 

kfold = KFold(5, shuffle=True, random_state=1)
accuracies, histories, losses = list(), list(), list()

for train_ix, test_ix in kfold.split(x_train, y_train):
    model = define_model()
    trainX, trainY, testX, testY = x_train[train_ix], y_train[train_ix], x_train[test_ix], y_train[test_ix]

    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    history = model.fit(trainX, trainY, batch_size=32, epochs=10, verbose=0, validation_data=(testX, testY))
    loss, accuracy = model.evaluate(testX, testY, verbose=0)

    accuracies.append(accuracy)
    histories.append(history)
    losses.append(loss)

#mean

print('losses: mean=%.3f std=%.3f, n=%d' % (mean(losses)*100, std(losses)*100, len(losses)))

#plot
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
