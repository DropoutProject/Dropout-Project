import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

sns.set(style="whitegrid")

def plot_feature_distributations(df, save=False):
    # Typo kept for compatibility with train.py
    for col in df.columns:
        if col == 'dropout_risk':
            continue
        plt.figure(figsize=(6, 4))
        if df[col].dtype == 'object':
            sns.countplot(x=col, hue='dropout_risk', data=df)
        else:
            sns.histplot(data=df, x=col, hue='dropout_risk', kde=True)
        plt.title(f'Distribution of {col} by dropout risk')
        if save:
            plt.savefig(f'{col}_distribution_by_dropout_risk.png', bbox_inches='tight')
        else:
            plt.show()

def plot_confusion(y_true, y_pred, save=False):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save:
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
    else:
        plt.show()

def plot_roc_curve(y_true, y_probs, save=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if save:
        plt.savefig('roc_curve.png', bbox_inches='tight')
    else:
        plt.show()

def plot_correlation_heatmap(df, save=False):
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    if save:
        plt.savefig('correlation_heatmap.png', bbox_inches='tight')
    else:
        plt.show()

def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)