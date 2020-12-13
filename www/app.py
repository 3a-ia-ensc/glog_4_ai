# -*- coding: utf-8 -*-

""" app.py
Create a web app to serve the model
"""
from json import dumps

from flask import Flask, request, jsonify, render_template, Response
import tensorflow as tf
import pickle

__author__ = "Simon Audrix and Gabriel Nativel-Fontaine"
__credits__ = ["Simon Audrix", "Gabriel Nativel-Fontaine"]
__copyright__ = "Copyright 2020}, Projet d'ingénierie logicielle pour l'IA"
__license__ = "WTFPL"
__version__ = "1.0.0"
__email__ = "gnativ910e@ensc.fr"
__status__ = "Development"


app = Flask(__name__)

@app.route('/')
def get_docs():
    """Index endpoint, display the documentation api
    """
    return render_template('swaggerui.html')

from urllib.parse import unquote
@app.route('/api/intent')
def _get_intent():
    """Makes a prediction on the received sentence as a parameter and returns the probability of belonging
    to each class of the model in json format.
    """
    to_send = request.args.get('sentence')
    MAX_SEQUENCE_LENGTH = 250

    with open('../models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    une_phrase = tokenizer.texts_to_sequences([to_send])
    une_phrase = tf.keras.preprocessing.sequence.pad_sequences(une_phrase, maxlen=MAX_SEQUENCE_LENGTH)

    model = tf.keras.models.load_model('../models/13_12_2020.hdf5', custom_objects=None, compile=True, options=None)
    prediction = model.predict(une_phrase)[0]

    labels = ['find-around-me', 'find-flight', 'find-hotel', 'find-restaurant',
              'find-train', 'irrelevant', 'provide-showtimes', 'purchase']

    dict_to_show = {}

    for i in range(8):
        dict_to_show[labels[i]] = str(prediction[i])

    response = app.response_class(response=dumps(dict_to_show),
                                  status=200,
                                  mimetype='application/json')
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == '__main__':
    app.run(debug=True)
