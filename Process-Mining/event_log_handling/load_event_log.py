"""
Event Log Loader
----------------
Simple functions to load and validate event log data.
"""

import pandas as pd


def validate_event_log(df):
    """Validate event log structure."""
    required_cols = ['case_id', 'activity', 'timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    return len(missing_cols) == 0


def standardize_columns(df):
    """Standardize common column names."""
    column_mapping = {
        'case': 'case_id',
        'caseid': 'case_id',
        'case id': 'case_id',
        'event': 'activity',
        'activity_name': 'activity',
        'time': 'timestamp',
        'datetime': 'timestamp',
        'date': 'timestamp'
    }
    
    df_renamed = df.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in df_renamed.columns:
            df_renamed = df_renamed.rename(columns={old_name: new_name})
    
    return df_renamed


def load_event_log(file_path):
    """Load event log from CSV file."""
    df = pd.read_csv(file_path)
    df = standardize_columns(df)
    
    # Try to parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    return df


def analyze_event_log(df):
    """Analyze basic event log properties."""
    analysis = {}
    
    if 'case_id' in df.columns:
        analysis['total_cases'] = df['case_id'].nunique()
        analysis['total_events'] = len(df)
    
    if 'activity' in df.columns:
        analysis['unique_activities'] = df['activity'].nunique()
        analysis['activity_frequency'] = df['activity'].value_counts()
    
    return analysis