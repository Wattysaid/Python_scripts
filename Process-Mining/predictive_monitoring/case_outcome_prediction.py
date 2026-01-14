"""
Case Outcome Prediction
----------------------
Functions for predicting case outcomes and remaining time.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

def predict_case_completion_time(partial_trace, historical_traces, case_durations):
    """Predict remaining time for a case based on current progress."""
    current_activities = set(partial_trace)
    current_length = len(partial_trace)
    
    # Find similar historical cases
    similar_cases = []
    for case_id, trace in historical_traces.items():
        if case_id in case_durations:
            # Check if current trace is a prefix of historical trace
            if len(trace) >= current_length:
                if trace[:current_length] == partial_trace:
                    remaining_activities = trace[current_length:]
                    total_duration = case_durations[case_id]
                    similar_cases.append({
                        'case_id': case_id,
                        'remaining_activities': remaining_activities,
                        'total_duration': total_duration,
                        'similarity': 1.0  # Exact prefix match
                    })
    
    if not similar_cases:
        # Find cases with similar activity patterns
        for case_id, trace in historical_traces.items():
            if case_id in case_durations:
                trace_activities = set(trace)
                similarity = len(current_activities.intersection(trace_activities)) / len(current_activities.union(trace_activities))
                
                if similarity > 0.5:  # Threshold for similarity
                    similar_cases.append({
                        'case_id': case_id,
                        'remaining_activities': [],
                        'total_duration': case_durations[case_id],
                        'similarity': similarity
                    })
    
    if not similar_cases:
        return {
            'predicted_remaining_time': None,
            'confidence': 0,
            'similar_cases_count': 0,
            'prediction_method': 'insufficient_data'
        }
    
    # Calculate prediction
    if similar_cases[0]['similarity'] == 1.0:  # Exact matches found
        exact_matches = [case for case in similar_cases if case['similarity'] == 1.0]
        remaining_durations = [case['total_duration'].total_seconds() / 3600 for case in exact_matches]
        predicted_remaining = np.mean(remaining_durations) * (1 - current_length / np.mean([len(historical_traces[case['case_id']]) for case in exact_matches]))
        confidence = min(0.9, len(exact_matches) / 10)  # Higher confidence for more exact matches
        method = 'exact_prefix_match'
    else:
        # Weighted average based on similarity
        weights = [case['similarity'] for case in similar_cases]
        durations = [case['total_duration'].total_seconds() / 3600 for case in similar_cases]
        predicted_remaining = np.average(durations, weights=weights) * 0.5  # Rough estimation
        confidence = min(0.7, len(similar_cases) / 20)
        method = 'similarity_based'
    
    return {
        'predicted_remaining_time_hours': max(0, predicted_remaining),
        'confidence': confidence,
        'similar_cases_count': len(similar_cases),
        'prediction_method': method,
        'similar_cases': similar_cases[:5]  # Top 5 for reference
    }

def predict_next_activities(partial_trace, historical_traces, top_n=5):
    """Predict most likely next activities."""
    next_activities = defaultdict(int)
    total_continuations = 0
    
    for case_id, trace in historical_traces.items():
        # Find all positions where partial_trace appears in trace
        for i in range(len(trace) - len(partial_trace) + 1):
            if trace[i:i+len(partial_trace)] == partial_trace:
                # Found a match, look at next activity
                if i + len(partial_trace) < len(trace):
                    next_activity = trace[i + len(partial_trace)]
                    next_activities[next_activity] += 1
                    total_continuations += 1
    
    if total_continuations == 0:
        return {
            'predictions': [],
            'confidence': 0,
            'total_patterns_found': 0
        }
    
    # Calculate probabilities
    predictions = []
    for activity, count in sorted(next_activities.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        probability = count / total_continuations
        predictions.append({
            'activity': activity,
            'probability': probability,
            'frequency': count
        })
    
    return {
        'predictions': predictions,
        'confidence': min(0.9, total_continuations / 50),
        'total_patterns_found': total_continuations
    }

def analyze_case_risk(partial_trace, historical_traces, case_durations, duration_threshold_hours=None):
    """Analyze risk factors for current case."""
    if duration_threshold_hours is None:
        all_durations = [d.total_seconds() / 3600 for d in case_durations.values()]
        duration_threshold_hours = np.percentile(all_durations, 75)  # 75th percentile as threshold
    
    current_length = len(partial_trace)
    current_activities = set(partial_trace)
    
    # Analyze historical patterns
    long_cases = []
    normal_cases = []
    
    for case_id, trace in historical_traces.items():
        if case_id in case_durations:
            duration_hours = case_durations[case_id].total_seconds() / 3600
            if duration_hours > duration_threshold_hours:
                long_cases.append({'case_id': case_id, 'trace': trace, 'duration': duration_hours})
            else:
                normal_cases.append({'case_id': case_id, 'trace': trace, 'duration': duration_hours})
    
    # Risk indicators
    risk_factors = []
    risk_score = 0
    
    # Factor 1: Current trace length vs normal
    normal_lengths = [len(case['trace']) for case in normal_cases]
    if normal_lengths:
        avg_normal_length = np.mean(normal_lengths)
        if current_length > avg_normal_length * 1.5:
            risk_factors.append('Unusually long process')
            risk_score += 0.3
    
    # Factor 2: Activities common in problematic cases
    if long_cases:
        long_case_activities = set()
        for case in long_cases:
            long_case_activities.update(case['trace'])
        
        problematic_activities = current_activities.intersection(long_case_activities)
        if problematic_activities:
            risk_factors.append(f'Contains activities common in long cases: {list(problematic_activities)}')
            risk_score += len(problematic_activities) / len(current_activities) * 0.4
    
    # Factor 3: Rework indicators (repeated activities)
    activity_counts = pd.Series(partial_trace).value_counts()
    rework_activities = activity_counts[activity_counts > 1]
    if not rework_activities.empty:
        risk_factors.append(f'Potential rework detected: {rework_activities.to_dict()}')
        risk_score += min(0.3, len(rework_activities) / len(current_activities))
    
    # Normalize risk score
    risk_score = min(1.0, risk_score)
    
    return {
        'risk_score': risk_score,
        'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.4 else 'Low',
        'risk_factors': risk_factors,
        'duration_threshold_hours': duration_threshold_hours,
        'current_trace_length': current_length,
        'analysis_metadata': {
            'long_cases_analyzed': len(long_cases),
            'normal_cases_analyzed': len(normal_cases),
            'avg_normal_case_length': np.mean(normal_lengths) if normal_lengths else 0
        }
    }