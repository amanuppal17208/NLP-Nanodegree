import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import normalize

np.random.seed(42)

tokenizer = Tokenizer(num_words=1000)

def get_model():

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(1000,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

if __name__=='__main__':

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000, skip_top=10)

    print(x_train.shape)
    print(x_test.shape)

    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
    x_train = normalize(x_train, norm='l2', axis=1)
    x_test = normalize(x_test, norm='l2', axis=1)
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(y_train.shape)
    print(y_test.shape)

    model=get_model()
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1)
    score = model.evaluate(x_test, y_test, verbose=0)

    print("Accuracy: ", score[1])