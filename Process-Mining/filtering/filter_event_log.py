"""
Event Log Filter
----------------
Filter event logs by time, activities, or cases.
"""

import pandas as pd
import sys

def ensure_packages():
    try:
        import pandas
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

def filter_by_time(df, start_date=None, end_date=None):
    if 'timestamp' not in df.columns:
        return df
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]
    
    return df

def filter_by_activities(df, activities):
    return df[df['activity'].isin(activities)]

def filter_by_case_count(df, min_events=None, max_events=None):
    case_counts = df['case_id'].value_counts()
    
    if min_events:
        valid_cases = case_counts[case_counts >= min_events].index
        df = df[df['case_id'].isin(valid_cases)]
    
    if max_events:
        valid_cases = case_counts[case_counts <= max_events].index
        df = df[df['case_id'].isin(valid_cases)]
    
    return df