import pandas as pd
import numpy as np

def load_data(file_path, file_type):
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'excel':
        return pd.read_excel(file_path)
    elif file_type == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def handle_missing_values(df, strategy='mean', fill_value=None):
    if strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'fill' and fill_value is not None:
        return df.fillna(fill_value)
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError(f"Invalid missing value strategy: {strategy}")

def remove_duplicates(df):
    return df.drop_duplicates()

def standardize_column_names(df):
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def adjust_data_types(df):
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column], errors='ignore')
        except:
            continue
    return df

def handle_outliers(df, z_threshold=3):
    for column in df.select_dtypes(include=[np.number]).columns:
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        df = df[z_scores.abs() <= z_threshold]
    return df

def clean_data(file_path, file_type='csv', missing_strategy='mean', drop_duplicates=True, fill_value=None):
    df = load_data(file_path, file_type)
    df = handle_missing_values(df, strategy=missing_strategy, fill_value=fill_value)
    if drop_duplicates:
        df = remove_duplicates(df)
    df = adjust_data_types(df)
    df = standardize_column_names(df)
    df = handle_outliers(df)
    return df