# -*- coding: utf-8 -*-

""" visualisations.py
This module is a specific module that allows to display visualisations under Plotly in order to analyse a dataset
and a predictive model.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import FloatProgress
from IPython.display import HTML, display
import tabulate

from src.api import predict

__author__ = "Simon Audrix and Gabriel Nativel-Fontaine"
__credits__ = ["Simon Audrix", "Gabriel Nativel-Fontaine"]
__copyright__ = "Copyright 2020}, Projet d'ingénierie logicielle pour l'IA"
__license__ = "WTFPL"
__version__ = "1.0.0"
__email__ = "gnativ910e@ensc.fr"
__status__ = "Development"


def histogram_intents(dataset: pd.DataFrame) -> None:
    """Displays the distribution of sentences in each intents

    Parameters:
    dataset (pandas.DataFrame): The dataframe containing intents and sentences
    """
    fig = px.histogram(dataset, x='intent', template='plotly_white', title='Sentence count by intent')
    fig.update_xaxes(categoryorder='total descending').update_yaxes(title='Number of sentences')
    fig.show()


def words_by_sentences(dataset: pd.DataFrame) -> None:
    """Displays the distribution of words in sentences

    Parameters:
    dataset (pandas.DataFrame): The dataframe containing intents and sentences
    """
    dataset['word_count'] = dataset['sentence'].str.findall(r'(\w+\'\w+)|(\w+)').str.len()

    fig = px.histogram(dataset, x='word_count', template='plotly_white', title='')
    fig.update_xaxes(categoryorder='total descending').update_yaxes(title='Number of sentences')
    fig.show()


def display_box_plot_nb_words(dataset: pd.DataFrame) -> None:
    """Displays the profil of words in intents

    Parameters:
    dataset (pandas.DataFrame): The dataframe containing intents and sentences
    """
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=dataset['word_count'],
        x=dataset['intent'],
        name='Mean & SD',
        marker_color='royalblue',
        boxmean='sd'  # represent mean and standard deviation
    ))

    fig.show()


def multiple_prediction(url_model: str, df_data: pd.DataFrame) -> tuple:
    """Call the API to get the model predictions for the data frame sentences.

    Parameters:
    url_model (str): API url
    df_data (pandas.DataFrame): The dataframe containing intents and sentences

    Returns:
    np array: The predicted values
    np array: The prediction pobabilities for each value
    """
    f = FloatProgress(min=0, max=len(df_data))
    display(f)

    predictions = []
    predictions_prob = []

    for i in range(len(df_data)):
        f.value += 1
        sentence = df_data['sentence'][i]

        pred = predict(url_model, sentence)
        max_val = max(pred, key=pred.get)
        predictions.append(max_val)
        predictions_prob.append(list(pred.values()))
        # prevent API overload
        if i % 1000 == 0:
            time.sleep(1)

    return np.array(predictions), np.array(predictions_prob)


def metrics_analysis(predictions: np.array, labels: np.array, cl:bool) -> None:
    """Calculates accuracy, precision, recall and F-Score on the predictions made by the model

    Parameters:
    predictions (array): Table of predictions
    labels (array): Table of real labels

    """
    if cl:
        classes = ['find-around-me', 'find-flight', 'find-hotel', 'find-restaurant',
                   'find-train', 'irrelevant', 'provide-showtimes', 'purchase']
    else:
        classes = ['find-train', 'irrelevant', 'find-flight', 'find-restaurant',
                   'purchase', 'find-around-me', 'provide-showtimes', 'find-hotel']

    confusion = np.zeros((len(classes), len(classes)))

    for i in range(len(classes)):
        for j in range(len(classes)):
            if not cl: confusion[j, i] = np.sum(((predictions == classes[i]) & (labels == classes[j])))
            else: confusion[j, i] = np.sum(((predictions == i) & (labels == j)))

    accuracy = np.diag(confusion).sum() / confusion.sum()

    precision = 0
    rappel = 0
    for i in range(len(classes)):
        precision += (confusion[i, i] / np.sum(confusion[:, i]))
        rappel += (confusion[i, i] / np.sum(confusion[i, :]))

    precision /= len(classes)
    rappel /= len(classes)

    f_score = 2 * ((precision * rappel) / (precision + rappel))

    table = []
    # display data
    for i in range(len(classes)):
        line = []
        for j in range(len(classes)):
            line.append(confusion[i, j])
        table.append(line)

    table = pd.DataFrame(table, index=classes)
    display(HTML(tabulate.tabulate(table, tablefmt='html', headers=classes)))

    metrics = pd.DataFrame([[accuracy], [precision], [rappel], [f_score]],
                           index=['Accuracy', 'Precision', 'Recall', 'F-Score'])
    display(HTML(tabulate.tabulate(metrics, tablefmt='html')))


def multiple_roc_curves(df_data: pd.DataFrame, predictions_prob: np.array, cl: bool) -> None:
    """Compute and display roc curve for each class

    Parameters:
    predictions_prob (array): Table of predictions (must contain probability for each class)

    """
    if not cl:
        classes = ['find-train', 'irrelevant', 'find-flight', 'find-restaurant',
                   'purchase', 'find-around-me', 'provide-showtimes', 'find-hotel']
    else:
        classes = ['find-around-me', 'find-flight', 'find-hotel', 'find-restaurant',
                   'find-train', 'irrelevant', 'provide-showtimes', 'purchase']

    df_results = pd.DataFrame()
    for i in range(len(classes)):
        cl = classes[i]
        df_results[cl] = df_data['intent'] == cl
        df_results['prob_' + cl] = predictions_prob[:, i]

    fpr = {}
    tpr = {}
    auc = {}
    thresh = {}

    for cl in classes:
        fpr[cl], tpr[cl], thresh[cl] = roc_curve(df_data['intent'], df_results['prob_' + cl], pos_label=cl)

        y_pred = df_results[cl]
        y_true = df_results['prob_' + cl] > 0.5
        auc[cl] = roc_auc_score(y_true, y_pred)

    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range(len(classes)):
        cl = classes[i]
        name = f"{cl} (AUC={auc[cl]:.2f})"
        fig.add_trace(go.Scatter(x=fpr[cl], y=tpr[cl], name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    fig.show()


def multiple_average_precision(df_data: pd.DataFrame, predictions_prob: np.array) -> None:
    """Compute and display precision-recall curve for each class

    Parameters:
    predictions_prob (array): Table of predictions (must contain probability for each class)

    """
    classes = ['find-train', 'irrelevant', 'find-flight', 'find-restaurant',
               'purchase', 'find-around-me', 'provide-showtimes', 'find-hotel']

    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )

    df_results = pd.DataFrame()
    for i in range(len(classes)):
        cl = classes[i]
        df_results[cl] = df_data['intent'] == cl
        df_results['prob_' + cl] = predictions_prob[:, i]

    prec = {}
    rec = {}
    auc = {}

    for cl in classes:
        y_true = df_results[cl]
        y_score = df_results['prob_' + cl]

        prec[cl], rec[cl], _ = precision_recall_curve(y_true, y_score)
        auc[cl] = average_precision_score(y_true, y_score)

    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for i in range(len(classes)):
        cl = classes[i]
        name = f"{cl} (AP={auc[cl]:.2f})"
        fig.add_trace(go.Scatter(x=rec[cl], y=prec[cl], name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    fig.show()


def wrong_predictions(predictions: np.array, prob: np.array, true_labels: np.array, data: pd.DataFrame) -> np.array:
    """Return the sentences with bad predictions
    """

    def set_class(x):
        classes = ['find-around-me', 'find-flight', 'find-hotel', 'find-restaurant',
                   'find-train', 'irrelevant', 'provide-showtimes', 'purchase']

        return classes[x]

    best_prob = np.max(prob, axis=1)
    wrong_idx = predictions != true_labels
    wrong_pred = data[wrong_idx]
    prob_wrong = best_prob[wrong_idx]
    wrong_labels = predictions[wrong_idx]
    wrong_labels = np.array(list(map(set_class, wrong_labels)))

    d = wrong_pred.copy()
    d['wrong'] = wrong_labels
    d['prob'] = prob_wrong

    idx = d['prob'] > 0.85

    return d[idx]
