import numpy as np
import pandas as pd


def correlation_score_fn(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    It is assumed that the predictions are not constant.
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values

    assert y_true.shape == y_pred.shape, print(f"{y_true.shape} vs {y_pred.shape}.")

    corr_sum = 0
    for i in range(len(y_true)):
        corr_sum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corr_sum / len(y_true)