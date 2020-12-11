# -*- coding: utf-8 -*-

""" model.py

"""

__author__ = "Simon Audrix and Gabriel Nativel-Fontaine"
__credits__ = ["Simon Audrix", "Gabriel Nativel-Fontaine"]
__copyright__ = "Copyright 2020}, Projet d'ingénierie logicielle pour l'IA"
__license__ = "WTFPL"
__version__ = "1.0.0"
__email__ = "gnativ910e@ensc.fr"
__status__ = "Development"

import tensorflow as tf
from tensorflow.keras.layers import Dense


class Model:
    def __init__(self, input_dim):
        self._model = self._build_model(input_dim)

    def _build_model(self, input_dim):
        """ Create a model

        Parameters:
        input_dim (int): dimension of the input
        """
        model = tf.keras.Sequential()
        model.add(Dense(100, input_dim=input_dim, activation='relu'))
        model.add(Dense(50, input_dim=input_dim, activation='relu'))
        model.add(Dense(8, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def save_model(self, name):
        """ Save the model to load it later

        Parameters:
        model (tf.keras.Model): model to save
        name (string): name to retrieve the model
        """
        self._model.save(f'../models/{name}')

    def load_model(self, name):
        """ Load a saved model

        Parameters:
        name (string): name of the model to retrieve
        """
        self._model = tf.keras.models.load_model(f'../models/{name}')

    def train(self, x_train, y_train, x_test, y_test, verbose=False):
        history = self._model.fit(x_train, y_train,
                                  epochs=100,
                                  verbose=verbose,
                                  validation_data=(x_test, y_test),
                                  batch_size=10)

    def evaluate(self, x, y):
        loss, accuracy = self._model.evaluate(x, y, verbose=False)
        print("Accuracy: {:.4f}".format(accuracy))

    def summary(self):
        print(self._model.summary())


"""
def logistic():
    max_features = 10000
    sequence_length = 250

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print("Accuracy:", score)
"""