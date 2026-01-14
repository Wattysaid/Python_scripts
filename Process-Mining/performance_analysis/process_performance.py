"""
Process Performance Analysis
----------------------------
Simple functions to analyze process performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def calculate_case_duration(df):
    """Calculate duration for each case."""
    case_durations = []
    
    for case_id, group in df.groupby('case_id'):
        if len(group) > 0:
            start_time = group['timestamp'].min()
            end_time = group['timestamp'].max()
            duration = (end_time - start_time).total_seconds() / 3600  # Convert to hours
            case_durations.append({
                'case_id': case_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': duration,
                'num_activities': len(group)
            })
    
    return pd.DataFrame(case_durations)


def calculate_activity_performance(df):
    """Calculate performance metrics for each activity."""
    activity_stats = []
    
    for activity, group in df.groupby('activity'):
        stats = {
            'activity': activity,
            'frequency': len(group),
            'unique_cases': group['case_id'].nunique(),
        }
        
        activity_stats.append(stats)
    
    return pd.DataFrame(activity_stats)


def calculate_throughput(df, time_unit='day'):
    """Calculate process throughput over time."""
    df_sorted = df.sort_values('timestamp')
    
    if time_unit == 'day':
        df_sorted['period'] = df_sorted['timestamp'].dt.date
    elif time_unit == 'week':
        df_sorted['period'] = df_sorted['timestamp'].dt.to_period('W')
    elif time_unit == 'month':
        df_sorted['period'] = df_sorted['timestamp'].dt.to_period('M')
    
    # Cases started per period
    cases_started = df_sorted.groupby(['period'])['case_id'].nunique().reset_index()
    cases_started.columns = ['period', 'cases_started']
    
    return cases_started