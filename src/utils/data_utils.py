"""
Utility functions for data loading, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import DATA_PATH, MONTH_ORDER, CATEGORICAL_COLUMNS


@st.cache_data(ttl=3600)
def load_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and perform initial preprocessing on the hotel bookings dataset.
    
    Parameters:
    -----------
    filepath : Path, optional
        Path to the Excel file. Defaults to DATA_PATH from config.
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed hotel bookings dataframe.
    
    Notes:
    ------
    This function is cached using Streamlit's cache_data decorator for performance.
    The cache expires after 1 hour (3600 seconds).
    """
    if filepath is None:
        filepath = DATA_PATH
    
    df = pd.read_excel(filepath)
    df = preprocess_data(df)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preprocessing steps on the raw dataset.
    
    Steps performed:
    1. Handle missing values
    2. Convert data types
    3. Create derived features
    4. Remove invalid entries
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe from Excel file.
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe.
    """
    df = df.copy()
    
    # Handle missing values
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    
    # Convert month to categorical with proper ordering
    df['arrival_date_month'] = pd.Categorical(
        df['arrival_date_month'], 
        categories=MONTH_ORDER, 
        ordered=True
    )
    
    # Create month number for plotting
    month_to_num = {month: i+1 for i, month in enumerate(MONTH_ORDER)}
    df['arrival_month_num'] = df['arrival_date_month'].map(month_to_num)
    
    # Create total stay duration
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    
    # Create total guests
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    
    # Create arrival date for time series analysis
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' + 
        df['arrival_month_num'].astype(str) + '-' + 
        df['arrival_date_day_of_month'].astype(str),
        format='%Y-%m-%d',
        errors='coerce'
    )
    
    # Revenue proxy (ADR * total stay)
    df['estimated_revenue'] = df['adr'] * df['total_stay']
    
    # Remove invalid entries (negative ADR)
    df = df[df['adr'] >= 0]
    
    # Create stay type category
    df['stay_type'] = pd.cut(
        df['total_stay'],
        bins=[0, 1, 3, 7, float('inf')],
        labels=['Night Stay', 'Short Stay', 'Week Stay', 'Extended Stay']
    )
    
    # Create lead time category
    df['lead_time_category'] = pd.cut(
        df['lead_time'],
        bins=[0, 7, 30, 90, 180, float('inf')],
        labels=['Last Minute', 'Short Term', 'Medium Term', 'Long Term', 'Very Long Term']
    )
    
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The hotel bookings dataframe.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing various summary statistics.
    """
    summary = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'date_range': {
            'start': df['arrival_date'].min(),
            'end': df['arrival_date'].max()
        },
        'missing_values': df.isnull().sum().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'hotel_types': df['hotel'].nunique(),
        'countries': df['country'].nunique(),
        'cancellation_rate': df['is_canceled'].mean() * 100,
        'avg_lead_time': df['lead_time'].mean(),
        'avg_adr': df['adr'].mean(),
        'avg_stay_duration': df['total_stay'].mean(),
        'total_revenue_estimate': df[df['is_canceled'] == 0]['estimated_revenue'].sum(),
    }
    
    # Hotel-specific stats
    for hotel in df['hotel'].unique():
        hotel_df = df[df['hotel'] == hotel]
        key = hotel.lower().replace(' ', '_')
        summary[f'{key}_count'] = len(hotel_df)
        summary[f'{key}_cancellation_rate'] = hotel_df['is_canceled'].mean() * 100
    
    return summary


def get_numerical_stats(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Get detailed statistics for a numerical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe.
    column : str
        Name of the numerical column.
    
    Returns:
    --------
    Dict[str, float]
        Dictionary containing statistical measures.
    """
    stats = {
        'count': df[column].count(),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        '25%': df[column].quantile(0.25),
        '50%': df[column].quantile(0.50),
        '75%': df[column].quantile(0.75),
        'max': df[column].max(),
        'skewness': df[column].skew(),
        'kurtosis': df[column].kurtosis(),
        'missing': df[column].isnull().sum(),
        'missing_pct': df[column].isnull().sum() / len(df) * 100
    }
    return stats


def get_categorical_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Get detailed statistics for a categorical column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe.
    column : str
        Name of the categorical column.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing categorical statistics.
    """
    value_counts = df[column].value_counts()
    stats = {
        'count': df[column].count(),
        'unique': df[column].nunique(),
        'top': value_counts.idxmax() if len(value_counts) > 0 else None,
        'top_freq': value_counts.max() if len(value_counts) > 0 else 0,
        'top_pct': (value_counts.max() / len(df) * 100) if len(value_counts) > 0 else 0,
        'missing': df[column].isnull().sum(),
        'missing_pct': df[column].isnull().sum() / len(df) * 100,
        'value_counts': value_counts.to_dict()
    }
    return stats


def filter_dataframe(
    df: pd.DataFrame,
    hotel_filter: Optional[List[str]] = None,
    year_filter: Optional[List[int]] = None,
    month_filter: Optional[List[str]] = None,
    cancellation_filter: Optional[int] = None
) -> pd.DataFrame:
    """
    Apply multiple filters to the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe.
    hotel_filter : List[str], optional
        List of hotel types to include.
    year_filter : List[int], optional
        List of years to include.
    month_filter : List[str], optional
        List of months to include.
    cancellation_filter : int, optional
        0 for not canceled, 1 for canceled, None for all.
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe.
    """
    filtered_df = df.copy()
    
    if hotel_filter:
        filtered_df = filtered_df[filtered_df['hotel'].isin(hotel_filter)]
    
    if year_filter:
        filtered_df = filtered_df[filtered_df['arrival_date_year'].isin(year_filter)]
    
    if month_filter:
        filtered_df = filtered_df[filtered_df['arrival_date_month'].isin(month_filter)]
    
    if cancellation_filter is not None:
        filtered_df = filtered_df[filtered_df['is_canceled'] == cancellation_filter]
    
    return filtered_df


def prepare_time_series_data(
    df: pd.DataFrame,
    freq: str = 'M',
    agg_column: str = 'is_canceled',
    agg_func: str = 'count'
) -> pd.DataFrame:
    """
    Prepare data for time series analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The hotel bookings dataframe.
    freq : str
        Frequency for resampling ('D', 'W', 'M', 'Q', 'Y').
    agg_column : str
        Column to aggregate.
    agg_func : str
        Aggregation function ('count', 'sum', 'mean').
    
    Returns:
    --------
    pd.DataFrame
        Time series dataframe with date index.
    """
    df = df.dropna(subset=['arrival_date'])
    df = df.set_index('arrival_date')
    
    if agg_func == 'count':
        ts_data = df.resample(freq)[agg_column].count()
    elif agg_func == 'sum':
        ts_data = df.resample(freq)[agg_column].sum()
    elif agg_func == 'mean':
        ts_data = df.resample(freq)[agg_column].mean()
    else:
        ts_data = df.resample(freq)[agg_column].count()
    
    return ts_data.to_frame()


def encode_categorical_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'label'
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Encode categorical features for machine learning.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    columns : List[str], optional
        Columns to encode. If None, uses CATEGORICAL_COLUMNS from config.
    method : str
        Encoding method: 'label', 'onehot', or 'frequency'.
    
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Dict]]
        Encoded dataframe and encoding mappings.
    """
    if columns is None:
        columns = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    
    df_encoded = df.copy()
    encodings = {}
    
    for col in columns:
        if col not in df_encoded.columns:
            continue
            
        if method == 'label':
            unique_vals = df_encoded[col].astype(str).unique()
            encoding = {val: i for i, val in enumerate(sorted(unique_vals))}
            df_encoded[col] = df_encoded[col].astype(str).map(encoding)
            encodings[col] = encoding
            
        elif method == 'frequency':
            freq_map = df_encoded[col].value_counts(normalize=True).to_dict()
            df_encoded[col] = df_encoded[col].map(freq_map)
            encodings[col] = freq_map
    
    return df_encoded, encodings
