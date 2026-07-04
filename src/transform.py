# src/transform.py
"""Preprocessing transformers: scaling and encoding."""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_fraud_preprocessor():
    """Create ColumnTransformer for fraud data."""
    num_cols = ['purchase_value', 'age', 'time_since_signup', 
                'hour_of_day', 'day_of_week', 'trans_1h', 'trans_24h']
    cat_cols = ['source', 'browser', 'sex', 'country']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    return preprocessor, num_cols, cat_cols

def create_credit_preprocessor():
    """Create ColumnTransformer for credit card data (only scaling)."""
    # For credit data, we only scale Time and Amount
    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), ['Time', 'Amount'])
    ], remainder='passthrough')  # V1-V28 remain as-is (already scaled)
    
    return preprocessor
