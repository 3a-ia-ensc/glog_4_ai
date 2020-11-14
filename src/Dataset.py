# -*- coding: utf-8 -*-

import pandas as pd

__author__ = "Simon Audrix and Gabriel Nativel-Fontaine"
__credits__ = ["Simon Audrix", "Gabriel Nativel-Fontaine"]
__copyright__ = "Copyright 2020}, Projet d'ingénierie logicielle pour l'IA"
__license__ = "WTFPL"
__version__ = "1.0.0"
__email__ = "gnativ910e@ensc.fr"
__status__ = "Development"


class Dataset:
    """ Tha dataset object is used to manage JSON dataset

    Args:
        path_train (str): path of the training set
        path_test:  (str): path of the testing set

    Attributes:
        _test (pandas.DataFrame): DataFrame containing testing set
        _train (pandas.DataFrame): DataFrame containing training set
        _data (pandas.DataFrame): DataFrame containing the full set
    """
    def __init__(self, path_train, path_test):
        self._test = pd.read_json(path_test)
        self._train = pd.read_json(path_train)
        self._data = pd.concat([self._train, self._test], keys=[range(0, self.__len__())])

        print(f'Loaded {self.__len__()} rows')

    @property
    def test(self):
        return self._test

    @property
    def train(self):
        return self._train

    @property
    def data(self):
        return self._data

    def __len__(self):
        return len(self._train) + len(self._test)
