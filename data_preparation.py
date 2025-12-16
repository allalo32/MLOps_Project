"""
Data Preparation Script for Traffic Volume Prediction
Handles data cleaning, preprocessing, and time-based splitting
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load the traffic volume dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    return df


def explore_data(df):
    """Explore and understand the dataset"""
    print("\n" + "="*80)
    print("DATA EXPLORATION")
    print("="*80)
    
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nUnique values in categorical columns:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].nunique()} unique values")
        if df[col].nunique() < 20:
            print(f"  Values: {df[col].unique()[:10]}")
    
    return df


def clean_data(df):
    """Clean the dataset"""
    print("\n" + "="*80)
    print("DATA CLEANING")
    print("="*80)
    
    df_clean = df.copy()
    
    # 1. Handle missing values
    print("\n1. Handling missing values...")
    missing_before = df_clean.isnull().sum().sum()
    print(f"   Total missing values before: {missing_before}")
    
    # Fill numeric columns with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"   Filled {col} with median: {median_val}")
    
    # Fill categorical columns with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'date_time' and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"   Filled {col} with mode: {mode_val}")
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"   Total missing values after: {missing_after}")
    
    # 2. Convert date_time to datetime format
    print("\n2. Converting date_time to datetime format...")
    df_clean['date_time'] = pd.to_datetime(df_clean['date_time'])
    print(f"   Date range: {df_clean['date_time'].min()} to {df_clean['date_time'].max()}")
    
    # 3. Normalize text fields
    print("\n3. Normalizing text fields...")
    text_cols = ['holiday', 'weather_main', 'weather_description']
    for col in text_cols:
        if col in df_clean.columns:
            # Convert to lowercase and strip whitespace
            df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
            print(f"   Normalized {col}")
    
    # 4. Handle outliers using IQR method
    print("\n4. Handling outliers in traffic_volume...")
    Q1 = df_clean['traffic_volume'].quantile(0.25)
    Q3 = df_clean['traffic_volume'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive outlier removal
    upper_bound = Q3 + 3 * IQR
    
    outliers_before = len(df_clean)
    df_clean = df_clean[(df_clean['traffic_volume'] >= lower_bound) & 
                        (df_clean['traffic_volume'] <= upper_bound)]
    outliers_removed = outliers_before - len(df_clean)
    print(f"   Removed {outliers_removed} outliers ({outliers_removed/outliers_before*100:.2f}%)")
    print(f"   Traffic volume range: {lower_bound:.0f} to {upper_bound:.0f}")
    
    # 5. Remove duplicates
    print("\n5. Removing duplicates...")
    duplicates_before = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = duplicates_before - df_clean.duplicated().sum()
    print(f"   Removed {duplicates_removed} duplicate rows")
    
    print(f"\nFinal dataset shape: {df_clean.shape}")
    
    return df_clean


def feature_engineering(df):
    """Create additional features from existing data"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df_fe = df.copy()
    
    # Extract time-based features
    print("\n1. Extracting time-based features...")
    df_fe['year'] = df_fe['date_time'].dt.year
    df_fe['month'] = df_fe['date_time'].dt.month
    df_fe['day'] = df_fe['date_time'].dt.day
    df_fe['hour'] = df_fe['date_time'].dt.hour
    df_fe['day_of_week'] = df_fe['date_time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_fe['is_weekend'] = (df_fe['day_of_week'] >= 5).astype(int)
    
    # Create rush hour feature
    df_fe['is_rush_hour'] = ((df_fe['hour'] >= 7) & (df_fe['hour'] <= 9) | 
                              (df_fe['hour'] >= 16) & (df_fe['hour'] <= 18)).astype(int)
    
    print(f"   Created features: year, month, day, hour, day_of_week, is_weekend, is_rush_hour")
    
    # Create holiday indicator (binary)
    print("\n2. Creating holiday indicator...")
    df_fe['is_holiday'] = (df_fe['holiday'] != 'none').astype(int)
    print(f"   Holiday percentage: {df_fe['is_holiday'].mean()*100:.2f}%")
    
    return df_fe


def encode_categorical(df):
    """Encode categorical variables"""
    print("\n" + "="*80)
    print("ENCODING CATEGORICAL VARIABLES")
    print("="*80)
    
    df_encoded = df.copy()
    
    # One-hot encode weather_main and weather_description
    print("\n1. One-hot encoding weather features...")
    
    # Weather main
    weather_main_dummies = pd.get_dummies(df_encoded['weather_main'], prefix='weather_main')
    df_encoded = pd.concat([df_encoded, weather_main_dummies], axis=1)
    print(f"   Created {len(weather_main_dummies.columns)} weather_main features")
    
    # Weather description (limit to top categories to avoid too many features)
    top_descriptions = df_encoded['weather_description'].value_counts().head(15).index
    df_encoded['weather_description_grouped'] = df_encoded['weather_description'].apply(
        lambda x: x if x in top_descriptions else 'other'
    )
    weather_desc_dummies = pd.get_dummies(df_encoded['weather_description_grouped'], 
                                          prefix='weather_desc')
    df_encoded = pd.concat([df_encoded, weather_desc_dummies], axis=1)
    print(f"   Created {len(weather_desc_dummies.columns)} weather_description features")
    
    # Drop original categorical columns (keep date_time for sorting)
    cols_to_drop = ['holiday', 'weather_main', 'weather_description', 'weather_description_grouped']
    df_encoded = df_encoded.drop(columns=cols_to_drop)
    
    print(f"\nFinal feature count: {len(df_encoded.columns)}")
    print(f"Features: {df_encoded.columns.tolist()}")
    
    return df_encoded


def time_based_split(df, train_pct=0.35, val_pct=0.35, test_pct=0.30):
    """
    Split data chronologically
    First 35% -> Training
    Next 35% -> Validation
    Final 30% -> Test
    """
    print("\n" + "="*80)
    print("TIME-BASED DATA SPLITTING")
    print("="*80)
    
    # Sort by date_time
    df_sorted = df.sort_values('date_time').reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    print(f"\nDataset size: {n} rows")
    print(f"\nTraining set:")
    print(f"  Size: {len(train_df)} rows ({len(train_df)/n*100:.1f}%)")
    print(f"  Date range: {train_df['date_time'].min()} to {train_df['date_time'].max()}")
    
    print(f"\nValidation set:")
    print(f"  Size: {len(val_df)} rows ({len(val_df)/n*100:.1f}%)")
    print(f"  Date range: {val_df['date_time'].min()} to {val_df['date_time'].max()}")
    
    print(f"\nTest set:")
    print(f"  Size: {len(test_df)} rows ({len(test_df)/n*100:.1f}%)")
    print(f"  Date range: {test_df['date_time'].min()} to {test_df['date_time'].max()}")
    
    # Verify no overlap
    assert train_df['date_time'].max() < val_df['date_time'].min(), "Train and validation overlap!"
    assert val_df['date_time'].max() < test_df['date_time'].min(), "Validation and test overlap!"
    print("\n[OK] No temporal overlap between splits")
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir='data'):
    """Save the train, validation, and test sets"""
    print("\n" + "="*80)
    print("SAVING DATA SPLITS")
    print("="*80)
    
    train_path = f"{output_dir}/train.csv"
    val_path = f"{output_dir}/validate.csv"
    test_path = f"{output_dir}/test.csv"
    
    train_df.to_csv(train_path, index=False)
    print(f"[OK] Saved training set to {train_path}")
    
    val_df.to_csv(val_path, index=False)
    print(f"[OK] Saved validation set to {val_path}")
    
    test_df.to_csv(test_path, index=False)
    print(f"[OK] Saved test set to {test_path}")
    
    print(f"\nAll splits saved successfully!")


def main():
    """Main execution function"""
    print("="*80)
    print("TRAFFIC VOLUME PREDICTION - DATA PREPARATION")
    print("="*80)
    
    # Load data
    df = load_data('data/Metro_Interstate_Traffic_Volume.csv')
    
    # Explore data
    df = explore_data(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Feature engineering
    df_fe = feature_engineering(df_clean)
    
    # Encode categorical variables
    df_encoded = encode_categorical(df_fe)
    
    # Time-based split
    train_df, val_df, test_df = time_based_split(df_encoded)
    
    # Save splits
    save_splits(train_df, val_df, test_df)
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the generated train.csv, validate.csv, and test.csv files")
    print("2. Proceed to AutoML analysis with H2O")


if __name__ == "__main__":
    main()
