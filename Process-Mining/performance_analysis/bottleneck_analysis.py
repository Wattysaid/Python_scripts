"""
Bottleneck Analysis
------------------
Functions for identifying and analyzing process bottlenecks.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta

def identify_activity_bottlenecks(event_log, time_col='timestamp', activity_col='activity', case_col='case_id'):
    """Identify bottleneck activities based on processing times."""
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(event_log[time_col]):
        event_log[time_col] = pd.to_datetime(event_log[time_col])
    
    activity_stats = {}
    
    # Calculate processing time for each activity
    for case_id in event_log[case_col].unique():
        case_events = event_log[event_log[case_col] == case_id].sort_values(time_col)
        
        for i in range(len(case_events) - 1):
            current_activity = case_events.iloc[i][activity_col]
            next_timestamp = case_events.iloc[i + 1][time_col]
            current_timestamp = case_events.iloc[i][time_col]
            
            processing_time = (next_timestamp - current_timestamp).total_seconds() / 3600  # hours
            
            if current_activity not in activity_stats:
                activity_stats[current_activity] = []
            activity_stats[current_activity].append(processing_time)
    
    # Analyze statistics for each activity
    bottleneck_analysis = {}
    
    for activity, times in activity_stats.items():
        if times:  # Only process activities with timing data
            mean_time = np.mean(times)
            median_time = np.median(times)
            std_time = np.std(times)
            max_time = max(times)
            min_time = min(times)
            q75 = np.percentile(times, 75)
            q95 = np.percentile(times, 95)
            
            bottleneck_analysis[activity] = {
                'mean_processing_time_hours': mean_time,
                'median_processing_time_hours': median_time,
                'std_processing_time_hours': std_time,
                'max_processing_time_hours': max_time,
                'min_processing_time_hours': min_time,
                'q75_processing_time_hours': q75,
                'q95_processing_time_hours': q95,
                'occurrences': len(times),
                'coefficient_of_variation': std_time / mean_time if mean_time > 0 else 0,
                'bottleneck_score': mean_time * len(times) * (std_time / mean_time if mean_time > 0 else 0)
            }
    
    # Rank activities by bottleneck potential
    sorted_activities = sorted(
        bottleneck_analysis.items(),
        key=lambda x: x[1]['bottleneck_score'],
        reverse=True
    )
    
    return {
        'activity_analysis': bottleneck_analysis,
        'bottleneck_ranking': sorted_activities,
        'top_bottlenecks': sorted_activities[:5],
        'total_activities_analyzed': len(bottleneck_analysis)
    }

def analyze_waiting_times(event_log, time_col='timestamp', activity_col='activity', case_col='case_id', resource_col='resource'):
    """Analyze waiting times between activities."""
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(event_log[time_col]):
        event_log[time_col] = pd.to_datetime(event_log[time_col])
    
    waiting_times = defaultdict(list)
    resource_utilization = defaultdict(list)
    
    for case_id in event_log[case_col].unique():
        case_events = event_log[event_log[case_col] == case_id].sort_values(time_col)
        
        for i in range(len(case_events) - 1):
            current_event = case_events.iloc[i]
            next_event = case_events.iloc[i + 1]
            
            # Calculate waiting time (gap between activities)
            waiting_time = (next_event[time_col] - current_event[time_col]).total_seconds() / 3600
            
            transition = f"{current_event[activity_col]} -> {next_event[activity_col]}"
            waiting_times[transition].append(waiting_time)
            
            # Resource utilization analysis
            if resource_col in event_log.columns:
                current_resource = current_event[resource_col]
                resource_utilization[current_resource].append({
                    'activity': current_event[activity_col],
                    'processing_time': waiting_time,
                    'timestamp': current_event[time_col]
                })
    
    # Analyze waiting time statistics
    waiting_analysis = {}
    for transition, times in waiting_times.items():
        if times:
            waiting_analysis[transition] = {
                'mean_waiting_time_hours': np.mean(times),
                'median_waiting_time_hours': np.median(times),
                'std_waiting_time_hours': np.std(times),
                'max_waiting_time_hours': max(times),
                'q90_waiting_time_hours': np.percentile(times, 90),
                'occurrences': len(times),
                'total_waiting_time_hours': sum(times)
            }
    
    # Resource utilization analysis
    resource_analysis = {}
    if resource_col in event_log.columns:
        for resource, activities in resource_utilization.items():
            processing_times = [act['processing_time'] for act in activities]
            if processing_times:
                resource_analysis[resource] = {
                    'total_processing_time_hours': sum(processing_times),
                    'avg_processing_time_hours': np.mean(processing_times),
                    'activities_handled': len(activities),
                    'unique_activities': len(set(act['activity'] for act in activities)),
                    'utilization_score': sum(processing_times) / len(activities)
                }
    
    return {
        'waiting_time_analysis': waiting_analysis,
        'resource_analysis': resource_analysis,
        'longest_waiting_transitions': sorted(
            waiting_analysis.items(),
            key=lambda x: x[1]['mean_waiting_time_hours'],
            reverse=True
        )[:10]
    }

def identify_process_bottlenecks(traces, case_durations, activity_frequencies=None):
    """Comprehensive bottleneck identification across the entire process."""
    # Analyze trace patterns
    trace_lengths = [len(trace) for trace in traces.values()]
    trace_patterns = Counter(' -> '.join(trace) for trace in traces.values())
    
    # Activity frequency analysis
    if activity_frequencies is None:
        activity_frequencies = Counter()
        for trace in traces.values():
            activity_frequencies.update(trace)
    
    # Duration analysis
    if case_durations:
        durations_hours = [d.total_seconds() / 3600 for d in case_durations.values()]
        duration_stats = {
            'mean_duration': np.mean(durations_hours),
            'median_duration': np.median(durations_hours),
            'q90_duration': np.percentile(durations_hours, 90),
            'max_duration': max(durations_hours)
        }
        
        # Identify long-duration cases
        long_duration_threshold = np.percentile(durations_hours, 75)
        long_cases = {case_id: trace for case_id, trace in traces.items() 
                     if case_id in case_durations and 
                     case_durations[case_id].total_seconds() / 3600 > long_duration_threshold}
    else:
        duration_stats = {}
        long_cases = {}
    
    # Analyze bottleneck patterns
    bottleneck_indicators = {}
    
    # 1. Activities that appear frequently in long cases
    if long_cases:
        long_case_activities = Counter()
        for trace in long_cases.values():
            long_case_activities.update(set(trace))  # Count unique activities per case
        
        total_long_cases = len(long_cases)
        bottleneck_activities = {}
        for activity, count in long_case_activities.items():
            frequency_in_long_cases = count / total_long_cases
            overall_frequency = activity_frequencies.get(activity, 0) / len(traces)
            
            if frequency_in_long_cases > overall_frequency * 1.5:  # 50% higher than normal
                bottleneck_activities[activity] = {
                    'frequency_in_long_cases': frequency_in_long_cases,
                    'overall_frequency': overall_frequency,
                    'bottleneck_ratio': frequency_in_long_cases / overall_frequency
                }
        
        bottleneck_indicators['problematic_activities'] = bottleneck_activities
    
    # 2. Process complexity analysis
    unique_traces = len(trace_patterns)
    total_cases = len(traces)
    process_complexity = {
        'variant_diversity': unique_traces / total_cases,
        'avg_trace_length': np.mean(trace_lengths),
        'std_trace_length': np.std(trace_lengths),
        'max_trace_length': max(trace_lengths),
        'complexity_score': (unique_traces / total_cases) * np.std(trace_lengths)
    }
    
    # 3. Rework analysis
    rework_cases = {}
    for case_id, trace in traces.items():
        activity_counts = Counter(trace)
        rework_activities = {act: count for act, count in activity_counts.items() if count > 1}
        if rework_activities:
            rework_cases[case_id] = {
                'trace': trace,
                'rework_activities': rework_activities,
                'total_rework': sum(count - 1 for count in rework_activities.values())
            }
    
    rework_analysis = {
        'cases_with_rework': len(rework_cases),
        'rework_percentage': len(rework_cases) / len(traces) * 100,
        'avg_rework_per_case': np.mean([case['total_rework'] for case in rework_cases.values()]) if rework_cases else 0
    }
    
    return {
        'duration_statistics': duration_stats,
        'process_complexity': process_complexity,
        'bottleneck_indicators': bottleneck_indicators,
        'rework_analysis': rework_analysis,
        'long_duration_cases': len(long_cases),
        'total_cases_analyzed': len(traces),
        'recommendations': generate_bottleneck_recommendations(
            bottleneck_indicators, process_complexity, rework_analysis
        )
    }

def generate_bottleneck_recommendations(bottleneck_indicators, process_complexity, rework_analysis):
    """Generate actionable recommendations based on bottleneck analysis."""
    recommendations = []
    
    # Activity-based recommendations
    if 'problematic_activities' in bottleneck_indicators:
        problematic_activities = bottleneck_indicators['problematic_activities']
        if problematic_activities:
            top_problematic = max(problematic_activities.items(), key=lambda x: x[1]['bottleneck_ratio'])
            recommendations.append({
                'type': 'activity_optimization',
                'priority': 'high',
                'description': f"Optimize '{top_problematic[0]}' activity - appears {top_problematic[1]['bottleneck_ratio']:.1f}x more in long cases",
                'activity': top_problematic[0],
                'impact': 'high'
            })
    
    # Process complexity recommendations
    if process_complexity['variant_diversity'] > 0.3:
        recommendations.append({
            'type': 'process_standardization',
            'priority': 'medium',
            'description': f"High process variability ({process_complexity['variant_diversity']:.1%}) suggests need for standardization",
            'impact': 'medium'
        })
    
    # Rework recommendations
    if rework_analysis['rework_percentage'] > 20:
        recommendations.append({
            'type': 'quality_improvement',
            'priority': 'high',
            'description': f"{rework_analysis['rework_percentage']:.1f}% of cases require rework - focus on quality improvement",
            'impact': 'high'
        })
    
    # Process length recommendations
    if process_complexity['avg_trace_length'] > 15:
        recommendations.append({
            'type': 'process_simplification',
            'priority': 'medium',
            'description': f"Average process length ({process_complexity['avg_trace_length']:.1f} activities) is high - consider simplification",
            'impact': 'medium'
        })
    
    return recommendations