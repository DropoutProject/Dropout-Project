import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed
from utils.preprocess import preprocess_data
from tensorflow.keras.models import save_model

# Reproducibility
set_random_seed(42)

# Load and preprocess data
df = pd.read_csv("student_data.csv")
X, y = preprocess_data(df, fit=True)
#save_encoders(encoders, scaler)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def build_model(trial):
    model = Sequential()
    input_dim = X_train.shape[1]

    # First layer
    model.add(Dense(trial.suggest_int("units1", 64, 256), input_dim=input_dim, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float("dropout1", 0.2, 0.5)))

    # Optional second hidden layer
    if trial.suggest_categorical("use_second_layer", [True, False]):
        model.add(Dense(trial.suggest_int("units2", 32, 128), activation="relu"))
        model.add(Dropout(trial.suggest_float("dropout2", 0.1, 0.4)))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    # Compile
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def objective(trial):
    model = build_model(trial)

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=50,
        batch_size=batch_size,
        verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    # Predict and evaluate
    preds = model.predict(X_valid).flatten()
    auc = roc_auc_score(y_valid, preds)
    return auc

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

print("Best parameters:", study.best_params)
print("Best AUC:", study.best_value)

# Train and save best model
best_params = study.best_params

# Manually build model using best_params

model = Sequential()
model.add(Dense(best_params['units1'], input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(best_params['dropout1']))

if best_params.get('use_second_layer', False):
    model.add(Dense(best_params['units2'], activation='relu'))
    model.add(Dropout(best_params['dropout2']))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=best_params['batch_size'],
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

model.save("model/nn_model.h5")
