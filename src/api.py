# -*- coding: utf-8 -*-

""" visualisations.py
This module is a specific module that allows to display visualisations under Plotly in order to analyse a dataset
and a predictive model.
"""

import requests


__author__ = "Simon Audrix and Gabriel Nativel-Fontaine"
__credits__ = ["Simon Audrix", "Gabriel Nativel-Fontaine"]
__copyright__ = "Copyright 2020}, Projet d'ingénierie logicielle pour l'IA"
__license__ = "WTFPL"
__version__ = "1.0.0"
__email__ = "gnativ910e@ensc.fr"
__status__ = "Development"


def api_request(path, params):
    query = ''
    for key, value in params.items():
        query += key + '=' + value + '&'

    # encoded_sentence = parse.quote(query[:-1], safe='')
    r = requests.get(f'{path}?{query}')
    return r


def predict(url, sentence):
    request = api_request(url, {'sentence': sentence})
    result = request.json()
    
    return result

