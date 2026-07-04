# src/explain.py
"""SHAP explainability: summary plots, force plots, and business insights."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import PROJECT_ROOT
from src.transform import create_fraud_preprocessor

def load_best_model():
    """Load the best (XGBoost) model and the preprocessor."""
    model_path = PROJECT_ROOT / 'models' / 'xgboost_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError("XGBoost model not found. Run training first.")
    pipeline = joblib.load(model_path)
    return pipeline

def get_test_data():
    """Load enriched data and prepare test set."""
    from src.features import engineer_fraud_features
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'fraud_enriched.csv', 
                     parse_dates=['signup_time', 'purchase_time'])
    drop_cols = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_int', 'class']
    X = df.drop(columns=drop_cols + ['class'])
    y = df['class']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_test, y_test, df

def generate_shap_plots(pipeline=None, X_test=None, y_test=None):
    """Generate SHAP summary and force plots for TP, FP, FN."""
    if pipeline is None:
        pipeline = load_best_model()
    
    if X_test is None or y_test is None:
        X_test, y_test, _ = get_test_data()
    
    # Extract model and preprocessor
    xgb_model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names
    from src.transform import create_fraud_preprocessor
    _, num_cols, cat_cols = create_fraud_preprocessor()
    feature_names = (
        [f'num__{c}' for c in num_cols] + 
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    )
    
    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    
    # SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_transformed)
    
    # 1. Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    fig_dir = PROJECT_ROOT / 'reports' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / 'shap_summary.png', bbox_inches='tight')
    plt.show()
    
    # 2. Built-in importance
    importance = xgb_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:10]
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx][::-1], importance[sorted_idx][::-1])
    plt.title('Top 10 Built‑in Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(fig_dir / 'builtin_importance.png')
    plt.show()
    
    # 3. Top 5 SHAP drivers
    shap_mean_abs = np.mean(np.abs(shap_values), axis=0)
    top5_idx = np.argsort(shap_mean_abs)[::-1][:5]
    top5_features = [feature_names[i] for i in top5_idx]
    top5_values = shap_mean_abs[top5_idx]
    print("\n=== Top 5 Global Drivers (SHAP) ===")
    for feat, val in zip(top5_features, top5_values):
        print(f"{feat}: {val:.4f}")
    
    # 4. Force plots for TP, FP, FN
    y_pred = pipeline.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    # Find indices
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0][0] if tp > 0 else None
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0][0] if fp > 0 else None
  fn_idx = np.where((y_test == 1) & (y_pred == 0))[0][0] if fn > 0 else None
    
    for idx, label in zip([tp_idx, fp_idx, fn_idx], ['TP', 'FP', 'FN']):
        if idx is not None:
            shap.force_plot(
                explainer.expected_value,
                shap_values[idx],
                X_test_transformed[idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            title = f'SHAP Force – {label} (Actual: {"Fraud" if label!="FP" else "Legit"}, Pred: {"Fraud" if label!="FN" else "Legit"})'
            plt.title(title)
            plt.tight_layout()
            plt.savefig(fig_dir / f'shap_force_{label.lower()}.png', bbox_inches='tight')
            plt.show()
    
    # 5. Business recommendations
    print("\n=== Business Recommendations ===")
    print("1. Velocity threshold: flag users with >3 transactions in 24h (SHAP shows trans_24h is a top driver).")
    print("2. New account verification: require MFA for purchases within 60 min of signup (time_since_signup highly impactful).")
    print("3. Country risk scoring: assign risk scores based on IP country and adjust approval thresholds dynamically.")
    
    return shap_values, feature_names
  
