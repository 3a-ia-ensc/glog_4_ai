
import pandas as pd
import numpy as np
import nltk
from tensorflow.keras.utils import to_categorical
from nltk import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpus/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextDataset:
    """ TextDataset object is used to manage text dataset
        It builds from json file

    Args:
        json_files (tuple): paths of the files to parse
        x_col (str): name of the column containing data
        y_col (str): name of the column containing labels

    Attributes:
        _data (pandas.DataFrame): DataFrame containing the full set
    """

    def __init__(self, json_files: str, x_col: str, y_col: str):
        self._data = self._read_json(json_files)
        self._labels = None

        self._add_one_hot()

        print(f'Loaded {self.__len__()} rows')

    def _read_json(self, tuple_files):
        """ Read multiple json files and concat them in a single DataFrame

        Parameters:
        tuple_files (tuple): path of the files
        """
        df = pd.DataFrame()

        for file in tuple_files:
            df = df.append(pd.read_json(file), ignore_index=True)

        return df

    def _add_one_hot(self):
        """ Add labels converted to one hot vector to the dataset
        """
        self._labels, indices = np.unique(self._data['intent'], return_inverse=True)
        one_hot_values = to_categorical(indices)

        self._data = pd.concat((self._data, pd.DataFrame(one_hot_values)), axis=1)

    def _find_synonyms(self, word):
        """ Find the french synonyms of a given word

        Parameters:
        word (str): a word
        """
        synonyms = []
        for synset in wordnet.synsets(word):
            for syn in synset.lemma_names('fra'):
                if syn not in synonyms:
                    synonyms.append(syn)

        return synonyms

    def _synonym_replacement(self, sentence):
        """ Build new sentenced by converting some words to there synonyms

        Parameters:
        sentence (str): a sentence
        """
        toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
        words = toknizer.tokenize(sentence)
        stoplist = stopwords.words('french')
        stoplist.append('ferret')
        n_sentence = []
        for w in words:
            if w not in stoplist:
                syn = self._find_synonyms(w)
                if len(syn) > 0:
                    for s in syn[:min(10, len(syn))]:
                        n_sentence.append(sentence.replace(w, s))

        return n_sentence

    def augment_data(self):
        """ Augment the dataset
        """
        new_sentences = []
        labels = []
        one_hot_lab = []
        for index, row in self._data.iterrows():
            if row['intent'] != 'irrelevant':
                sentences = self._synonym_replacement(row['sentence'])
                for s in sentences:
                    new_sentences.append(s)
                    labels.append(row['intent'])
                    vector = np.zeros(8)
                    idx = list(self._labels).index(row['intent'])
                    vector[idx] = 1
                    one_hot_lab.append(vector)

        new_data = pd.DataFrame({'sentence': new_sentences, 'intent': labels})
        ones = pd.DataFrame(one_hot_lab)
        return pd.concat((new_data, ones), axis=1)

    def augment_and_balance(self):
        """ Augment and balance the dataset, it takes the smallest number of occurence
        of one classe and balance the number in other classes
        """
        self._data = self._data.sample(frac=1)
        augmented_data = self.augment_data().sample(frac=1)

        # counts
        count_init = self._data['intent'].value_counts()
        count_augm = augmented_data['intent'].value_counts()
        count_augm['irrelevant'] = 0

        sum_counts = count_init + count_augm
        min_value = min(sum_counts)

        n_df = pd.DataFrame()

        for cl in self._labels:
            if count_init[cl] >= min_value:
                select = self._data.loc[self._data['intent'] == cl][:min_value]
                n_df = n_df.append(select, ignore_index=True)
            else:
                missing_data = min_value - count_init[cl]
                n_df = n_df.append(self._data.loc[self._data['intent'] == cl], ignore_index=True)
                select = augmented_data.loc[augmented_data['intent'] == cl][:missing_data]
                n_df = n_df.append(select, ignore_index=True)

        balanced_data = n_df.sample(frac=1)
        balanced_data['intent'].value_counts()
        self._data = balanced_data
        print(f'Dataset contains now {self.__len__()} rows')

    def split_data(self, frac=0.2):
        """ Split the dataset into training set and testing set

        Parameters:
        frac (double): the fraction of dataset to be used as test set
        """
        df = self._data.sample(frac=1)
        size_train = int((1 - frac) * self.__len__())
        return df[:size_train], df[size_train:]

    @property
    def data(self):
        return self._data

    def __len__(self):
        return len(self._data)
