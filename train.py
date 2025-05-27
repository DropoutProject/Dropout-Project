import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from utils.preprocess import preprocess_data
from utils.visualize import (
    plot_feature_distributations,
    plot_roc_curve,
    plot_confusion,
    plot_correlation_heatmap,
    print_classification_report,
)
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load dataset
df = pd.read_csv("student_data.csv")

# Preprocess
X, y = preprocess_data(df, fit=True)
#save_encoders(encoders, scaler)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ========== Train XGBoost ==========
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False,
)
joblib.dump(xgb_model, "model/xgb_model.pkl")

# ========== Train Neural Network ==========
nn_model = Sequential([
    Dense(128, activation="relu", input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])

nn_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

nn_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
    verbose=1
)
nn_model.save("model/nn_model.h5")

# ========== Evaluate Models ==========
print("\n--- XGBoost Evaluation ---")
xgb_preds = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, xgb_preds))
print("Classification Report:\n", classification_report(y_test, xgb_preds))

print("\n--- Neural Network Evaluation ---")
nn_model = load_model("model/nn_model.h5")
nn_preds = nn_model.predict(X_test).flatten()
nn_preds_binary = (nn_preds > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, nn_preds_binary))
print("Classification Report:\n", classification_report(y_test, nn_preds_binary))

# ========== Visualizations ==========
os.makedirs("figures", exist_ok=True)
plot_feature_distributations(df, save=True)
plot_correlation_heatmap(df, save=True)

# Ensembling predictions
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
nn_probs = nn_model.predict(X_test).flatten()
avg_probs = (xgb_probs + nn_probs) / 2
final_preds = (avg_probs > 0.5).astype(int)

plot_confusion(y_test, final_preds, save=True)
plot_roc_curve(y_test, avg_probs, save=True)
print_classification_report(y_test, final_preds)
