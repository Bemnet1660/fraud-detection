"""
Task 1: Data Understanding and Preprocessing
- Cleans data, adds geolocation, engineers features, applies SMOTE.
- Saves cleaned datasets and train/test splits (pkl).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. Load data
# ------------------------------
print("Loading data...")
fraud_df = pd.read_csv('Fraud_Data.csv')
credit_df = pd.read_csv('creditcard.csv')
ip_country_df = pd.read_csv('IpAddress_to_Country.csv')

# ------------------------------
# 2. Clean Fraud_Data
# ------------------------------
print("Cleaning Fraud_Data...")
# No missing values; remove duplicates
fraud_df.drop_duplicates(inplace=True)
# Convert types
fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
fraud_df['class'] = fraud_df['class'].astype(int)

def ip_to_int(ip):
    parts = ip.split('.')
    return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])

fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)

# ------------------------------
# 3. Clean creditcard.csv
# ------------------------------
print("Cleaning creditcard.csv...")
credit_df.drop_duplicates(inplace=True)
credit_df['Class'] = credit_df['Class'].astype(int)

# ------------------------------
# 4. EDA (basic plots saved)
# ------------------------------
print("Generating EDA plots...")
fig, axes = plt.subplots(1,2, figsize=(12,4))
fraud_balance = fraud_df['class'].value_counts(normalize=True)
credit_balance = credit_df['Class'].value_counts(normalize=True)
sns.barplot(x=fraud_balance.index, y=fraud_balance.values, ax=axes[0])
axes[0].set_title('Fraud_Data - Class (1=Fraud)')
sns.barplot(x=credit_balance.index, y=credit_balance.values, ax=axes[1])
axes[1].set_title('Creditcard - Class (1=Fraud)')
plt.tight_layout()
plt.savefig('class_imbalance.png')
plt.close()

# ------------------------------
# 5. Geolocation enrichment (Fraud_Data only)
# ------------------------------
print("Adding geolocation...")
def ip_range_to_int(ip_str):
    parts = ip_str.split('.')
    return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])

ip_country_df['lower_int'] = ip_country_df['lower_bound_ip_address'].apply(ip_range_to_int)
ip_country_df['upper_int'] = ip_country_df['upper_bound_ip_address'].apply(ip_range_to_int)
ip_country_df.sort_values('lower_int', inplace=True)

fraud_df_sorted = fraud_df.sort_values('ip_int')
merged = pd.merge_asof(
    fraud_df_sorted,
    ip_country_df[['lower_int', 'upper_int', 'country']],
    left_on='ip_int',
    right_on='lower_int',
    direction='forward'
)
merged = merged[merged['ip_int'] <= merged['upper_int']]
merged.drop(['ip_int', 'lower_int', 'upper_int'], axis=1, inplace=True)
fraud_df = merged.sort_index()

# ------------------------------
# 6. Feature engineering (Fraud_Data)
# ------------------------------
print("Engineering features...")
fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
fraud_df['time_since_signup_hours'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600

fraud_df = fraud_df.sort_values(['user_id', 'purchase_time'])
fraud_df['user_transaction_count'] = fraud_df.groupby('user_id')['user_id'].transform('count')
fraud_df['time_diff_prev_purchase_hours'] = fraud_df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
fraud_df['time_diff_prev_purchase_hours'].fillna(999, inplace=True)
user_avg_gap = fraud_df.groupby('user_id')['time_diff_prev_purchase_hours'].mean()
fraud_df['user_avg_gap_hours'] = fraud_df['user_id'].map(user_avg_gap)
fraud_df['device_usage_count'] = fraud_df.groupby('device_id')['device_id'].transform('count')
fraud_df['ip_usage_count'] = fraud_df.groupby('ip_address')['ip_address'].transform('count')
fraud_df.reset_index(drop=True, inplace=True)

# Save cleaned & featured e‑commerce data
fraud_df.to_csv('Fraud_Data_cleaned_featured.csv', index=False)
credit_df.to_csv('creditcard_cleaned.csv', index=False)

# ------------------------------
# 7. Data transformation & SMOTE
# ------------------------------
print("Transforming data and applying SMOTE...")
# ---------- Fraud_Data ----------
cat_features = ['source', 'browser', 'sex', 'country']
num_features = ['purchase_value', 'age', 'hour_of_day', 'day_of_week',
                'time_since_signup_hours', 'user_transaction_count',
                'time_diff_prev_purchase_hours', 'user_avg_gap_hours',
                'device_usage_count', 'ip_usage_count']

X_fraud = fraud_df[cat_features + num_features]
y_fraud = fraud_df['class']

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fraud, y_fraud, test_size=0.2, stratify=y_fraud, random_state=42
)

preprocessor_fraud = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ])

pipeline_fraud = Pipeline(steps=[
    ('preprocessor', preprocessor_fraud),
    ('smote', SMOTE(random_state=42))
])
X_train_f_res, y_train_f_res = pipeline_fraud.fit_resample(X_train_f, y_train_f)
X_test_f_trans = pipeline_fraud.named_steps['preprocessor'].transform(X_test_f)

# ---------- creditcard ----------
X_credit = credit_df.drop('Class', axis=1)
y_credit = credit_df['Class']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_credit, y_credit, test_size=0.2, stratify=y_credit, random_state=42
)
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)
smote_c = SMOTE(random_state=42)
X_train_c_res, y_train_c_res = smote_c.fit_resample(X_train_c_scaled, y_train_c)

# ------------------------------
# 8. Save train/test sets as .pkl
# ------------------------------
print("Saving train/test sets (pkl)...")
train_test_data = {
    'fraud': {
        'X_train': X_train_f_res,
        'y_train': y_train_f_res,
        'X_test': X_test_f_trans,
        'y_test': y_test_f
    },
    'credit': {
        'X_train': X_train_c_res,
        'y_train': y_train_c_res,
        'X_test': X_test_c_scaled,
        'y_test': y_test_c
    }
}
joblib.dump(train_test_data, 'train_test_sets.pkl')

print("Task 1 completed successfully!")
print("Outputs: Fraud_Data_cleaned_featured.csv, creditcard_cleaned.csv, train_test_sets.pkl, class_imbalance.png")
