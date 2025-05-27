import optuna
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from utils.preprocess import preprocess_data

# Load and preprocess data
df = pd.read_csv("student_data.csv")
X, y = preprocess_data(df, fit=True)
#save_encoders(encoders, scaler)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
              early_stopping_rounds=10, verbose=False)

    preds = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    return auc

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
print("Best AUC:", study.best_value)

# Train final model on best parameters
best_model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric="logloss")
best_model.fit(X_train, y_train)
joblib.dump(best_model, "model/xgb_model.pkl")
