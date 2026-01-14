"""
Simple Process Discovery
------------------------
Simple functions to discover process models from event logs.
"""

import pandas as pd
from collections import defaultdict, Counter


def extract_traces(df):
    """Extract process traces from event log."""
    if 'timestamp' in df.columns:
        df = df.sort_values(['case_id', 'timestamp'])
    
    traces = df.groupby('case_id')['activity'].apply(list).to_dict()
    return traces


def find_directly_follows_relations(traces):
    """Find directly-follows relationships between activities."""
    df_relations = defaultdict(int)
    
    for case_id, trace in traces.items():
        for i in range(len(trace) - 1):
            current_activity = trace[i]
            next_activity = trace[i + 1]
            df_relations[(current_activity, next_activity)] += 1
    
    return dict(df_relations)


def discover_start_end_activities(traces):
    """Find start and end activities."""
    start_activities = Counter()
    end_activities = Counter()
    
    for trace in traces.values():
        if trace:
            start_activities[trace[0]] += 1
            end_activities[trace[-1]] += 1
    
    return dict(start_activities), dict(end_activities)


def generate_process_model(df_relations, start_activities, end_activities, min_frequency=1):
    """Generate a simple process model."""
    # Filter relations by minimum frequency
    filtered_relations = {k: v for k, v in df_relations.items() if v >= min_frequency}
    
    model = {
        'activities': set(),
        'relations': filtered_relations,
        'start_activities': start_activities,
        'end_activities': end_activities
    }
    
    # Extract all activities
    for (source, target), freq in filtered_relations.items():
        model['activities'].add(source)
        model['activities'].add(target)
    
    return model