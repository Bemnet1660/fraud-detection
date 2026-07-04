# src/data_loader.py
"""Module for loading raw data files."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(file).parent.parent

def load_fraud_data(raw: bool = True) -> pd.DataFrame:
    """Load e-commerce fraud dataset.
    
    Args:
        raw: If True, load from data/raw/, else from data/processed/.
    
    Returns:
        DataFrame with Fraud_Data.csv
    """
    if raw:
        path = PROJECT_ROOT / 'data' / 'raw' / 'Fraud_Data.csv'
    else:
        path = PROJECT_ROOT / 'data' / 'processed' / 'fraud_cleaned.csv'
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Parse dates if it's the raw file
    if raw:
        df = pd.read_csv(path)
        # Dates will be parsed later in clean step
    else:
        df = pd.read_csv(path, parse_dates=['signup_time', 'purchase_time'])
    
    return df

def load_ip_map() -> pd.DataFrame:
    """Load IP-to-country mapping."""
    path = PROJECT_ROOT / 'data' / 'raw' / 'IpAddress_to_Country.csv'
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def load_credit_data(raw: bool = True) -> pd.DataFrame:
    """Load credit card transaction dataset."""
    if raw:
        path = PROJECT_ROOT / 'data' / 'raw' / 'creditcard.csv'
    else:
        path = PROJECT_ROOT / 'data' / 'processed' / 'credit_cleaned.csv'
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)
