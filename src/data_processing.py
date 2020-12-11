# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

__author__ = "Simon Audrix and Gabriel Nativel-Fontaine"
__credits__ = ["Simon Audrix", "Gabriel Nativel-Fontaine"]
__copyright__ = "Copyright 2020}, Projet d'ingénierie logicielle pour l'IA"
__license__ = "WTFPL"
__version__ = "1.0.0"
__email__ = "gnativ910e@ensc.fr"
__status__ = "Development"


def balance(data):
    """Balance the data

    Parameters:
    data (pandas.DataFrame): dataset
    """
    nb_irrelevant = 312  # nombre moyens de phrases par intent (sans les irrelevants)
    dataset_size = len(data.data[data.data.intent != 'irrelevant']) + 312

    not_irrelevant = data.data[data.data.intent != 'irrelevant']
    irrelevants = data.data[data.data.intent == 'irrelevant']
    ech_irrelevants = irrelevants.sample(frac=nb_irrelevant / len(irrelevants))

    dataset = pd.concat([not_irrelevant, ech_irrelevants], ignore_index=True)
    sample_dataset = dataset.sample(frac=1)

    return sample_dataset


def to_one_hot(data):
    """ Convert data to one hot vectors
    """
    indices = np.unique(data, return_inverse=True)[1]
    return to_categorical(indices)


def cut_data(sentences, labels, frac=0.25):
    """ Split data into train set and test set

    Parameters:
    data (pandas.DataFrame): dataset
    frac (float): fraction of split
    """
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=frac)
    return sentences_train, sentences_test, y_train, y_test


def vectorize_data(data):
    """ Split data into train set and test set
    Parameters:
    data (pandas.DataFrame): dataset
    """
    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    return vectorizer.transform(data)
