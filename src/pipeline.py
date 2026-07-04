# src/pipeline.py
"""End‑to‑end orchestration pipeline."""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from src.data_loader import load_fraud_data, load_ip_map, load_credit_data
from src.preprocess import clean_fraud_data, clean_credit_data, merge_ip_to_country
from src.features import engineer_fraud_features, save_enriched_data
from src.transform import create_fraud_preprocessor
from src.train import train_baseline, train_ensemble, get_fraud_data
from src.evaluate import evaluate_model, plot_confusion_matrix, compare_models
from src.explain import generate_shap_plots

def run_full_pipeline():
    """Execute the entire fraud detection pipeline."""
    print("=" * 60)
    print("FRAUD DETECTION PIPELINE – ADEY INNOVATIONS")
    print("=" * 60)
    
    # Step 1: Data Loading & Cleaning (Fraud)
    print("\n1. Loading and cleaning e‑commerce data...")
    fraud_raw = load_fraud_data(raw=True)
    fraud_clean = clean_fraud_data(fraud_raw)
    
    # Step 2: IP Geolocation
    print("\n2. Merging IP addresses with country mapping...")
    ip_map = load_ip_map()
    fraud_enriched = merge_ip_to_country(fraud_clean, ip_map)
    
    # Step 3: Feature Engineering
    print("\n3. Engineering temporal and velocity features...")
    fraud_feat = engineer_fraud_features(fraud_enriched)
    save_enriched_data(fraud_feat)
    
    # Step 4: Credit Card Data (simple cleaning)
    print("\n4. Cleaning credit card data...")
    credit_raw = load_credit_data(raw=True)
    credit_clean = clean_credit_data(credit_raw)
    credit_clean.to_csv('data/processed/credit_cleaned.csv', index=False)
    
    # Step 5: Prepare X, y for fraud
    print("\n5. Preparing features and target...")
    drop_cols = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_int', 'class']
    X = fraud_feat.drop(columns=drop_cols + ['class'])
    y = fraud_feat['class']
    
    # Step 6: Train Baseline
    print("\n6. Training Logistic Regression baseline...")
    baseline_pipeline, baseline_metrics, X_test, y_test, y_pred_lr, y_proba_lr = train_baseline(X, y, save_model=True)
    evaluate_model(y_test, y_pred_lr, y_proba_lr, "Logistic Regression")
    
    # Step 7: Train Ensemble
    print("\n7. Training XGBoost with tuning...")
    ensemble_pipeline, ensemble_metrics, X_test, y_test, y_pred_xgb, y_proba_xgb = train_ensemble(X, y, tune=True, save_model=True)
    evaluate_model(y_test, y_pred_xgb, y_proba_xgb, "XGBoost (Tuned)")
    
    # Step 8: Compare Models
    compare_models(baseline_metrics, ensemble_metrics)
    
    # Step 9: SHAP Explainability
    print("\n8. Generating SHAP plots...")
    generate_shap_plots(pipeline=ensemble_pipeline, X_test=X_test, y_test=y_test)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("Check reports/figures/ for visualizations.")
    print("=" * 60)

if name == "main":
    run_full_pipeline()
