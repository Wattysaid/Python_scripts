# Resource Utilization Analysis
import pandas as pd
import numpy as np
import sys

def calculate_resource_metrics(df):
    if 'resource' not in df.columns:
        print("No resource column found")
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate metrics per resource
    resource_stats = []
    
    for resource, group in df.groupby('resource'):
        cases = group['case_id'].nunique()
        activities = len(group)
        unique_activities = group['activity'].nunique() if 'activity' in group.columns else 0
        
        # Time span
        time_span = (group['timestamp'].max() - group['timestamp'].min()).total_seconds() / 3600  # hours
        
        # Activity rate
        activity_rate = activities / max(time_span, 1) if time_span > 0 else 0
        
        stats = {
            'resource': resource,
            'total_activities': activities,
            'unique_cases': cases,
            'unique_activity_types': unique_activities,
            'time_span_hours': time_span,
            'activity_rate_per_hour': activity_rate
        }
        
        resource_stats.append(stats)
    
    return pd.DataFrame(resource_stats)

def find_workload_distribution(resource_stats):
    if resource_stats is None or resource_stats.empty:
        return None
    
    total_activities = resource_stats['total_activities'].sum()
    resource_stats['workload_percentage'] = (resource_stats['total_activities'] / total_activities) * 100
    
    return resource_stats.sort_values('workload_percentage', ascending=False)