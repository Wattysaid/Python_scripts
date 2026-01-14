"""
Anomaly Detection
----------------
Functions for detecting anomalies in process execution.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy import stats

def detect_duration_anomalies(case_durations, method='zscore', threshold=3):
    """Detect cases with anomalous durations."""
    durations_hours = [d.total_seconds() / 3600 for d in case_durations.values()]
    case_ids = list(case_durations.keys())
    
    anomalies = []
    
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(durations_hours))
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        for idx in anomaly_indices:
            anomalies.append({
                'case_id': case_ids[idx],
                'duration_hours': durations_hours[idx],
                'z_score': z_scores[idx],
                'anomaly_type': 'duration_outlier',
                'severity': 'high' if z_scores[idx] > threshold * 1.5 else 'medium'
            })
    
    elif method == 'iqr':
        Q1 = np.percentile(durations_hours, 25)
        Q3 = np.percentile(durations_hours, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        for i, (case_id, duration) in enumerate(zip(case_ids, durations_hours)):
            if duration < lower_bound or duration > upper_bound:
                anomalies.append({
                    'case_id': case_id,
                    'duration_hours': duration,
                    'expected_range': f'{lower_bound:.2f} - {upper_bound:.2f}',
                    'anomaly_type': 'duration_outlier',
                    'severity': 'high' if duration > upper_bound + IQR else 'medium'
                })
    
    return {
        'anomalies': anomalies,
        'total_cases': len(case_durations),
        'anomaly_rate': len(anomalies) / len(case_durations) * 100,
        'detection_method': method,
        'statistics': {
            'mean_duration': np.mean(durations_hours),
            'std_duration': np.std(durations_hours),
            'median_duration': np.median(durations_hours)
        }
    }

def detect_sequence_anomalies(traces, min_support=0.05):
    """Detect unusual activity sequences."""
    # Extract all activity pairs (bigrams)
    activity_pairs = defaultdict(int)
    total_pairs = 0
    
    for case_id, trace in traces.items():
        for i in range(len(trace) - 1):
            pair = (trace[i], trace[i + 1])
            activity_pairs[pair] += 1
            total_pairs += 1
    
    # Calculate support for each pair
    pair_support = {pair: count / total_pairs for pair, count in activity_pairs.items()}
    
    # Find anomalous sequences in each trace
    anomalous_cases = []
    
    for case_id, trace in traces.items():
        anomaly_score = 0
        unusual_sequences = []
        
        for i in range(len(trace) - 1):
            pair = (trace[i], trace[i + 1])
            support = pair_support.get(pair, 0)
            
            if support < min_support:
                unusual_sequences.append({
                    'sequence': pair,
                    'position': i,
                    'support': support
                })
                anomaly_score += (min_support - support)
        
        if unusual_sequences:
            anomalous_cases.append({
                'case_id': case_id,
                'anomaly_score': anomaly_score,
                'unusual_sequences': unusual_sequences,
                'trace_length': len(trace),
                'anomaly_type': 'unusual_sequence'
            })
    
    return {
        'anomalous_cases': sorted(anomalous_cases, key=lambda x: x['anomaly_score'], reverse=True),
        'total_cases_analyzed': len(traces),
        'anomaly_rate': len(anomalous_cases) / len(traces) * 100,
        'min_support_threshold': min_support,
        'total_activity_pairs': len(activity_pairs),
        'statistics': {
            'avg_anomaly_score': np.mean([case['anomaly_score'] for case in anomalous_cases]) if anomalous_cases else 0,
            'max_anomaly_score': max([case['anomaly_score'] for case in anomalous_cases]) if anomalous_cases else 0
        }
    }

def detect_resource_anomalies(event_log, resource_col='resource', case_col='case_id', activity_col='activity'):
    """Detect unusual resource behavior patterns."""
    # Calculate resource utilization patterns
    resource_activities = defaultdict(list)
    resource_cases = defaultdict(set)
    activity_resources = defaultdict(set)
    
    for _, event in event_log.iterrows():
        resource = event[resource_col]
        case_id = event[case_col]
        activity = event[activity_col]
        
        resource_activities[resource].append(activity)
        resource_cases[resource].add(case_id)
        activity_resources[activity].add(resource)
    
    anomalous_patterns = []
    
    # Detect resources performing unusual activities
    for activity, resources in activity_resources.items():
        if len(resources) > 1:
            # Check for resources that rarely perform this activity
            activity_counts = Counter([event[resource_col] for _, event in event_log.iterrows() 
                                     if event[activity_col] == activity])
            total_activity_occurrences = sum(activity_counts.values())
            
            for resource, count in activity_counts.items():
                frequency = count / total_activity_occurrences
                if frequency < 0.1 and count < 3:  # Rare performer
                    anomalous_patterns.append({
                        'type': 'rare_activity_performer',
                        'resource': resource,
                        'activity': activity,
                        'frequency': frequency,
                        'count': count,
                        'severity': 'medium'
                    })
    
    # Detect resources with unusual workload
    resource_workloads = {resource: len(cases) for resource, cases in resource_cases.items()}
    
    if resource_workloads:
        workload_values = list(resource_workloads.values())
        q75 = np.percentile(workload_values, 75)
        q25 = np.percentile(workload_values, 25)
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        lower_bound = max(0, q25 - 1.5 * iqr)
        
        for resource, workload in resource_workloads.items():
            if workload > upper_bound:
                anomalous_patterns.append({
                    'type': 'high_workload',
                    'resource': resource,
                    'case_count': workload,
                    'expected_range': f'{lower_bound:.1f} - {upper_bound:.1f}',
                    'severity': 'high' if workload > upper_bound * 1.5 else 'medium'
                })
            elif workload < lower_bound and lower_bound > 0:
                anomalous_patterns.append({
                    'type': 'low_workload',
                    'resource': resource,
                    'case_count': workload,
                    'expected_range': f'{lower_bound:.1f} - {upper_bound:.1f}',
                    'severity': 'medium'
                })
    
    # Detect resources with unusual activity diversity
    resource_diversity = {resource: len(set(activities)) for resource, activities in resource_activities.items()}
    
    if resource_diversity:
        diversity_values = list(resource_diversity.values())
        avg_diversity = np.mean(diversity_values)
        std_diversity = np.std(diversity_values)
        
        for resource, diversity in resource_diversity.items():
            z_score = abs(diversity - avg_diversity) / std_diversity if std_diversity > 0 else 0
            
            if z_score > 2:  # Unusual diversity
                anomalous_patterns.append({
                    'type': 'unusual_activity_diversity',
                    'resource': resource,
                    'activity_count': diversity,
                    'z_score': z_score,
                    'average_diversity': avg_diversity,
                    'severity': 'high' if z_score > 3 else 'medium'
                })
    
    return {
        'anomalous_patterns': anomalous_patterns,
        'total_resources': len(resource_activities),
        'resource_statistics': {
            'avg_cases_per_resource': np.mean(list(resource_workloads.values())) if resource_workloads else 0,
            'avg_activities_per_resource': np.mean(list(resource_diversity.values())) if resource_diversity else 0,
            'total_unique_activities': len(activity_resources)
        }
    }