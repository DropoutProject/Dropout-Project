import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["parental_education", "income_level", "extracurriculars"]
numerical_features = [
    "attendance_rate",
    "GPA",
    "study_hours_per_week",
    "age",
    "failed_courses",
]


def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def preprocess_data(df, preprocessor=None, fit=False):
    df = df.copy()
    if fit:
        preprocessor = build_preprocessor()
        features = df.drop("dropout_risk", axis=1)
        target = df["dropout_risk"]
        X = preprocessor.fit_transform(features)
        joblib.dump(preprocessor, "model/encoders.pkl")
        return X, target
    else:
        features = df
        if "dropout_risk" in df.columns:
            features = df.drop("dropout_risk", axis=1)
        preprocessor = joblib.load("model/encoders.pkl")
        X = preprocessor.transform(features)
        return X

