# -*- coding: utf-8 -*-

""" app.py
Create a web app to serve the model
"""
import threading
import time
from queue import Empty, Queue
from json import dumps
import numpy as np
import flask
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

BATCH_SIZE = 20
BATCH_TIMEOUT = 0.5
CHECK_INTERVAL = 0.01

requests_queue = Queue()

def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (
                len(requests_batch) > BATCH_SIZE or
                (len(requests_batch) > 0 and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

        batch_inputs = np.empty([0, 250])
        for request in requests_batch:
            batch_inputs = np.concatenate((batch_inputs, request['input']), axis=0)

        #print(batch_inputs.shape)
        batch_outputs = model.predict(batch_inputs)
        #print(batch_inputs.shape)
        for request, output in zip(requests_batch, batch_outputs):
            request['output'] = output


threading.Thread(target=handle_requests_by_batch).start()

app = Flask(__name__)
model = tf.keras.models.load_model('models/best_model', custom_objects=None, compile=True, options=None)

@tf.function
def predict(une_phrase):
    prediction = model.predict(une_phrase)[0]

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
    to_send = flask.request.args.get('sentence')
    MAX_SEQUENCE_LENGTH = 250

    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    une_phrase = tokenizer.texts_to_sequences([to_send])
    une_phrase = tf.keras.preprocessing.sequence.pad_sequences(une_phrase, maxlen=MAX_SEQUENCE_LENGTH)
    #print(une_phrase.shape)
    request = {'input': une_phrase, 'time': time.time()}
    requests_queue.put(request)

    while 'output' not in request:
        time.sleep(CHECK_INTERVAL)

    # prediction = model.predict(une_phrase)[0]

    labels = ['find-around-me', 'find-flight', 'find-hotel', 'find-restaurant',
              'find-train', 'irrelevant', 'provide-showtimes', 'purchase']

    dict_to_show = {}
    prediction = request['output']

    for i in range(8):
        dict_to_show[labels[i]] = str(prediction[i])

    response = app.response_class(response=dumps(dict_to_show),
                                  status=200,
                                  mimetype='application/json')
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response



#if __name__ == '__main__':
#    app.run(debug=True)
