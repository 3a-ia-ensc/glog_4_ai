import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import time
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100


class ModelCustom:
    """ Model

    Attributes:
        _model (tf.keras.Model): The model
    """

    def __init__(self):
        self._model = self._buildModel()
        self._tokenizer = None

    def _buildModel(self) -> tf.keras.Model:
        """ Read multiple json files and concat them in a single DataFrame

        Parameters:
        tuple_files (tuple): path of the files

        Returns:
        Model : outputs a keras model to be used in production
        """

        input_model = Input(MAX_SEQUENCE_LENGTH)
        embed = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_model)
        drop = SpatialDropout1D(0.2)(embed)
        lstm = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(drop)
        out = Dense(8, activation='softmax')(lstm)

        model = tf.keras.Model(input_model, out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self, X:pd.DataFrame, Y:pd.DataFrame) -> tf.keras.callbacks.History:
        """ Train the model

        Parameters:
        X (pd.DataFrame): sentences to use for training
        Y (pd.DataFrame): labels to use for training

        Returns:
        keras hystory: Return the logs and data from the model training
        """
        self._tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        self._tokenizer.fit_on_texts(X)

        X = self._tokenizer.texts_to_sequences(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

        epochs = 15
        batch_size = 64
        checkpoint = tf.keras.callbacks.ModelCheckpoint('../models/best_model',
                                                        monitor='val_accuracy',
                                                        save_best_only=True,
                                                        verbose=0)
        history = self._model.fit(X, Y,
                                  verbose=2,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_split=0.1,
                                  callbacks=[
                                    EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001),
                                    checkpoint
                                  ])

        with open('../models/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self._tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self, X:pd.DataFrame, Y:pd.DataFrame):
        """ Train the model

        Parameters:
        X (pd.DataFrame): sentences to use for evaluation
        Y (pd.DataFrame): labels to use for evaluation
        """
        X = self._tokenizer.texts_to_sequences(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

        self._model.evaluate(X, Y)

    def summary(self):
        """ Display the model's summary
        """
        print(self._model.summary())

    def save(self):
        """ Save the model
        """
        self._model.save(f'../models/model_{time.time()}.hdf5', overwrite=True, include_optimizer=True, save_format='h5')

    def predict(self, X:pd.DataFrame) -> np.array:
        """ Make a prediction

        Parameters:
        X (pd.DataFrame): sentences to use for prediction

        Returns:
        np.array: outpus the array of probability for prediction classes
        """
        X = self._tokenizer.texts_to_sequences(X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

        return self._model.predict(X)
