import numpy as np


def ensemble_predict(xgb, nn, X, threshold=0.5):
    xgb_probs = xgb.predict_proba(X)[:, 1]
    nn_probs = nn.predict(X).flatten()
    avg_probs = (xgb_probs + nn_probs) / 2
    preds = (avg_probs >= threshold).astype(int)
    return preds, avg_probs
