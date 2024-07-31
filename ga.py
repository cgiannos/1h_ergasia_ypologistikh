import keras
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from keras.utils import np_utils
from matplotlib import pyplot as plt, pyplot
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from deepevolution import wrap_keras


wrap_keras()

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
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# fit_evolve genetic algorithm

accuracies, histories, losses = list(), list(), list()


model = define_model()

history = model.fit_evolve(x_train, y_train, max_generations=5, population=200, top_k=3, mutation_rate=0.1, mutation_std=0.01)
print(f"Model accuracy: {model.evaluate(x_train, y_train, batch_size=36)[1]}")

pd.DataFrame(history).plot()
plt.show()

