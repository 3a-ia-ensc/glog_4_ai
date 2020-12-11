# -*- coding: utf-8 -*-

""" app.py
Create a web app to serve the model
"""
from json import dumps

from flask import Flask, request, jsonify, render_template, Response

from src.data_processing import vectorize_data
from src.model import load_model

__author__ = "Simon Audrix and Gabriel Nativel-Fontaine"
__credits__ = ["Simon Audrix", "Gabriel Nativel-Fontaine"]
__copyright__ = "Copyright 2020}, Projet d'ingénierie logicielle pour l'IA"
__license__ = "WTFPL"
__version__ = "1.0.0"
__email__ = "gnativ910e@ensc.fr"
__status__ = "Development"


app = Flask(__name__)
model = load_model('simple_model')


@app.route('/')
def get_docs():
    """Index endpoint, display the documentation api
    """
    return render_template('swaggerui.html')


@app.route('/api/intent')
def _get_intent():
    """Makes a prediction on the received sentence as a parameter and returns the probability of belonging
    to each class of the model in json format.
    """
    to_send = request.args.get('sentence')

    vec = vectorize_data([to_send])
    prediction = model.predict([vec])
    """prediction = model.predict([to_send])
    print(prediction)
    to_send =  jsonify({"intent": prediction[0]})"""

    response = app.response_class(response=dumps({'sentence': to_send}),
                                  status=200,
                                  mimetype='application/json')
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == '__main__':
    app.run(debug=True)
