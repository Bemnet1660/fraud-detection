# src/features.py
"""Feature engineering: temporal, velocity, and geolocation-derived features."""

import pandas as pd
import numpy as np

def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time_since_signup, hour_of_day, day_of_week, and velocity features."""
    df = df.copy()
    
    # Ensure datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Time since signup (hours)
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    # Hour and day
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Velocity: transactions per user in 1h and 24h windows
    df = df.sort_values(['user_id', 'purchase_time'])
    
    # Use rolling with time-based windows
    # For each user, count previous transactions in window
    df['trans_1h'] = df.groupby('user_id')['purchase_time'].transform(
        lambda x: x.rolling('1H').count() - 1
    )
    df['trans_24h'] = df.groupby('user_id')['purchase_time'].transform(
        lambda x: x.rolling('24H').count() - 1
    )
    
    # Fill NaN (first transaction per user) with 0
    df['trans_1h'] = df['trans_1h'].fillna(0).astype(int)
    df['trans_24h'] = df['trans_24h'].fillna(0).astype(int)
    
    return df

def save_enriched_data(df: pd.DataFrame) -> None:
    """Save enriched dataset to processed folder."""
    from src.data_loader import PROJECT_ROOT
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / 'fraud_enriched.csv', index=False)
    print("Enriched dataset saved to data/processed/fraud_enriched.csv")
