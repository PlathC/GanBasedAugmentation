import os
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from typing import Dict, List


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues) -> None:
    """
    Confusion matrix plot helper function
    Args:
        cm: Confusion matrix
        classes: Classes in the order of the confusion matrix
        normalize: Normalise confusion matrix
        title: Plot title
        cmap: Color map
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.rcParams["figure.figsize"] = (20, 20)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, range(len(classes)), rotation=45)
    plt.yticks(tick_marks, range(len(classes)))

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def metrics(y_true: np.array, y_pred: np.array, classes: List) -> Dict:
    """
    Metrics generator function
    Args:
        y_true: Ground truth
        y_pred: Predicted values
        classes: Class names

    Returns:
        A dictionary containing the metrics and the confusion matrix
    """
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, normalize=True)
    return {
        'cm': cm,
        'macro': {
            'precision': sklearn.metrics.precision_score(y_true, y_pred, average='macro'),
            'recall': sklearn.metrics.recall_score(y_true, y_pred, average='macro'),
            'f1': sklearn.metrics.f1_score(y_true, y_pred, average='macro'),
        },
        'micro': {
            'precision': sklearn.metrics.precision_score(y_true, y_pred, average='micro'),
            'recall': sklearn.metrics.recall_score(y_true, y_pred, average='micro'),
            'f1': sklearn.metrics.f1_score(y_true, y_pred, average='micro'),
        },
        'MCC': sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    }
