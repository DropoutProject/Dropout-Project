import joblib
from tensorflow.keras.models import load_model


def load_models():
    xgb = joblib.load("model/xgb_model.pkl")
    nn = load_model("model/nn_model.h5")
    return xgb, nn


def load_preprocessor():
    return joblib.load("model/encoders.pkl")
