# src/preprocess.py
"""Data cleaning and IP geolocation integration."""

import pandas as pd
import numpy as np
from src.data_loader import PROJECT_ROOT

def ip_to_int(ip: str) -> int:
    """Convert dotted IPv4 address to integer."""
    parts = ip.split('.')
    return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])

def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Fraud_Data.csv: handle missing, duplicates, data types."""
    df = df.copy()
    
    # Missing values
    if 'age' in df.columns:
        df['age'].fillna(df['age'].median(), inplace=True)
    if 'sex' in df.columns:
        df['sex'].fillna(df['sex'].mode()[0], inplace=True)
    
    # Duplicates
    df.drop_duplicates(inplace=True)
    
    # Data types
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['class'] = df['class'].astype(int)
    
    return df

def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean creditcard.csv: drop duplicates."""
    df = df.copy()
    df.drop_duplicates(inplace=True)
    # Ensure Class is int
    df['Class'] = df['Class'].astype(int)
    return df

def merge_ip_to_country(fraud_df: pd.DataFrame, ip_map_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich fraud data with country from IP ranges using merge_asof."""
    df = fraud_df.copy()
    ip_map = ip_map_df.copy()
    
    # Convert IPs to integers
    df['ip_int'] = df['ip_address'].apply(ip_to_int)
    ip_map['lower_int'] = ip_map['lower_bound_ip_address'].apply(ip_to_int)
    ip_map['upper_int'] = ip_map['upper_bound_ip_address'].apply(ip_to_int)
    
    # Sort for merge_asof
    ip_map_sorted = ip_map.sort_values('lower_int')
    df_sorted = df.sort_values('ip_int')
    
    # Range lookup
    merged = pd.merge_asof(df_sorted, ip_map_sorted, left_on='ip_int', right_on='lower_int')
    # Filter to ensure ip_int <= upper_int
    merged = merged[merged['ip_int'] <= merged['upper_int']].copy()
    
    # Drop temporary IP columns (keep country)
    merged.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address', 
                         'lower_int', 'upper_int', 'ip_address'], inplace=True, errors='ignore')
    
    return merged

def save_cleaned_data(fraud_df: pd.DataFrame, credit_df: pd.DataFrame) -> None:
    """Save cleaned datasets to processed folder."""
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    fraud_df.to_csv(processed_dir / 'fraud_cleaned.csv', index=False)
    credit_df.to_csv(processed_dir / 'credit_cleaned.csv', index=False)
    print("Cleaned datasets saved to data/processed/")
