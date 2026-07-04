# src/evaluate.py
"""Evaluation utilities: metrics, confusion matrix, comparison."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix, classification_report
from src.data_loader import PROJECT_ROOT

def evaluate_model(y_true, y_pred, y_proba, model_name="Model"):
    """Compute and print evaluation metrics."""
    f1 = f1_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_proba)
    conf = confusion_matrix(y_true, y_pred)
    
    print(f"=== {model_name} ===")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print("\nConfusion Matrix:")
    print(conf)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {'f1': f1, 'auprc': auprc, 'confusion': conf}

def plot_confusion_matrix(conf, model_name="Model"):
    """Plot confusion matrix and save figure."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix – {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    fig_dir = PROJECT_ROOT / 'reports' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f'confusion_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

def compare_models(baseline_metrics, ensemble_metrics):
    """Print side-by-side comparison."""
    print("\n=== Model Comparison ===")
    print("Metric        | Logistic Regression | XGBoost (Tuned)")
    print(f"F1-Score      | {baseline_metrics['f1']:.4f}           | {ensemble_metrics['f1']:.4f}")
    print(f"AUPRC         | {baseline_metrics['auprc']:.4f}           | {ensemble_metrics['auprc']:.4f}")
    
    if 'cv_mean' in ensemble_metrics:
        print(f"CV AUPRC      | N/A                 | {ensemble_metrics['cv_mean']:.4f} ± {ensemble_metrics['cv_std']:.4f}")
