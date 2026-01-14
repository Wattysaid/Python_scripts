"""
Process Reports
--------------
Functions for generating comprehensive process mining reports.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import json

def generate_process_summary_report(traces, case_durations, event_log=None):
    """Generate comprehensive process summary report."""
    # Basic statistics
    total_cases = len(traces)
    total_activities = sum(len(trace) for trace in traces.values())
    unique_activities = len(set(activity for trace in traces.values() for activity in trace))
    
    # Duration statistics
    if case_durations:
        durations_hours = [d.total_seconds() / 3600 for d in case_durations.values()]
        duration_stats = {
            'mean_duration_hours': np.mean(durations_hours),
            'median_duration_hours': np.median(durations_hours),
            'std_duration_hours': np.std(durations_hours),
            'min_duration_hours': min(durations_hours),
            'max_duration_hours': max(durations_hours),
            'q1_duration_hours': np.percentile(durations_hours, 25),
            'q3_duration_hours': np.percentile(durations_hours, 75)
        }
    else:
        duration_stats = {}
    
    # Trace length statistics
    trace_lengths = [len(trace) for trace in traces.values()]
    trace_stats = {
        'mean_trace_length': np.mean(trace_lengths),
        'median_trace_length': np.median(trace_lengths),
        'std_trace_length': np.std(trace_lengths),
        'min_trace_length': min(trace_lengths),
        'max_trace_length': max(trace_lengths)
    }
    
    # Activity frequency
    activity_counts = Counter()
    for trace in traces.values():
        activity_counts.update(trace)
    
    top_activities = dict(activity_counts.most_common(10))
    
    # Variant analysis
    variant_patterns = Counter()
    for trace in traces.values():
        pattern = ' → '.join(trace)
        variant_patterns[pattern] += 1
    
    unique_variants = len(variant_patterns)
    top_variants = dict(variant_patterns.most_common(5))
    
    # Process complexity metrics
    complexity_metrics = {
        'variant_diversity': unique_variants / total_cases,
        'activity_reuse': total_activities / unique_activities,
        'avg_activities_per_case': total_activities / total_cases
    }
    
    # Resource statistics (if available)
    resource_stats = {}
    if event_log is not None and 'resource' in event_log.columns:
        unique_resources = event_log['resource'].nunique()
        resource_workload = event_log['resource'].value_counts()
        resource_stats = {
            'total_resources': unique_resources,
            'avg_events_per_resource': event_log.shape[0] / unique_resources,
            'most_active_resource': resource_workload.index[0],
            'resource_distribution': resource_workload.head().to_dict()
        }
    
    report = {
        'report_generated_at': datetime.now().isoformat(),
        'process_overview': {
            'total_cases': total_cases,
            'total_events': total_activities,
            'unique_activities': unique_activities,
            'unique_variants': unique_variants
        },
        'duration_analysis': duration_stats,
        'trace_analysis': trace_stats,
        'activity_analysis': {
            'top_activities': top_activities,
            'activity_distribution': dict(activity_counts)
        },
        'variant_analysis': {
            'top_variants': top_variants,
            'total_variants': unique_variants
        },
        'complexity_metrics': complexity_metrics,
        'resource_analysis': resource_stats
    }
    
    return report

def generate_performance_report(case_durations, event_log, time_col='timestamp', case_col='case_id'):
    """Generate detailed performance analysis report."""
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(event_log[time_col]):
        event_log[time_col] = pd.to_datetime(event_log[time_col])
    
    # Case duration analysis
    durations_hours = [d.total_seconds() / 3600 for d in case_durations.values()]
    
    duration_analysis = {
        'basic_statistics': {
            'mean': np.mean(durations_hours),
            'median': np.median(durations_hours),
            'std': np.std(durations_hours),
            'min': min(durations_hours),
            'max': max(durations_hours)
        },
        'percentiles': {
            'p10': np.percentile(durations_hours, 10),
            'p25': np.percentile(durations_hours, 25),
            'p75': np.percentile(durations_hours, 75),
            'p90': np.percentile(durations_hours, 90),
            'p95': np.percentile(durations_hours, 95)
        },
        'outlier_analysis': {
            'q1': np.percentile(durations_hours, 25),
            'q3': np.percentile(durations_hours, 75),
            'iqr': np.percentile(durations_hours, 75) - np.percentile(durations_hours, 25)
        }
    }
    
    # Add outlier counts
    q1, q3 = duration_analysis['outlier_analysis']['q1'], duration_analysis['outlier_analysis']['q3']
    iqr = duration_analysis['outlier_analysis']['iqr']
    outliers = [d for d in durations_hours if d < q1 - 1.5*iqr or d > q3 + 1.5*iqr]
    duration_analysis['outlier_analysis']['outlier_count'] = len(outliers)
    duration_analysis['outlier_analysis']['outlier_percentage'] = len(outliers) / len(durations_hours) * 100
    
    # Throughput analysis
    daily_starts = event_log.groupby(case_col)[time_col].min().dt.date.value_counts().sort_index()
    daily_completions = event_log.groupby(case_col)[time_col].max().dt.date.value_counts().sort_index()
    
    throughput_analysis = {
        'daily_averages': {
            'avg_starts_per_day': daily_starts.mean(),
            'avg_completions_per_day': daily_completions.mean(),
            'max_starts_per_day': daily_starts.max(),
            'max_completions_per_day': daily_completions.max()
        },
        'period_analysis': {
            'analysis_start_date': daily_starts.index.min().isoformat(),
            'analysis_end_date': daily_starts.index.max().isoformat(),
            'total_days': len(daily_starts.index),
            'total_cases_started': daily_starts.sum(),
            'total_cases_completed': daily_completions.sum()
        }
    }
    
    # Activity-level performance
    activity_durations = defaultdict(list)
    
    # Calculate time between consecutive activities for each case
    for case_id in event_log[case_col].unique():
        case_events = event_log[event_log[case_col] == case_id].sort_values(time_col)
        
        for i in range(len(case_events) - 1):
            current_activity = case_events.iloc[i]['activity']
            next_timestamp = case_events.iloc[i + 1][time_col]
            current_timestamp = case_events.iloc[i][time_col]
            
            duration_hours = (next_timestamp - current_timestamp).total_seconds() / 3600
            activity_durations[current_activity].append(duration_hours)
    
    activity_performance = {}
    for activity, durations in activity_durations.items():
        if durations:  # Only include activities with duration data
            activity_performance[activity] = {
                'mean_duration_hours': np.mean(durations),
                'median_duration_hours': np.median(durations),
                'std_duration_hours': np.std(durations),
                'min_duration_hours': min(durations),
                'max_duration_hours': max(durations),
                'occurrences': len(durations)
            }
    
    report = {
        'report_generated_at': datetime.now().isoformat(),
        'case_duration_analysis': duration_analysis,
        'throughput_analysis': throughput_analysis,
        'activity_performance': activity_performance,
        'performance_summary': {
            'avg_case_duration_hours': np.mean(durations_hours),
            'fastest_case_hours': min(durations_hours),
            'slowest_case_hours': max(durations_hours),
            'total_cases_analyzed': len(case_durations)
        }
    }
    
    return report

def export_report_to_json(report, filename):
    """Export report to JSON file."""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return f"Report exported to {filename}"

def export_report_to_html(report, filename, title="Process Mining Report"):
    """Export report to HTML file."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
            h3 {{ color: #999; }}
            .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p><strong>Generated:</strong> {report.get('report_generated_at', 'Unknown')}</p>
    """
    
    # Process Overview Section
    if 'process_overview' in report:
        html_content += """
        <div class="section">
            <h2>Process Overview</h2>
        """
        for key, value in report['process_overview'].items():
            html_content += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value:,}</div>'
        html_content += "</div>"
    
    # Duration Analysis
    if 'case_duration_analysis' in report:
        html_content += """
        <div class="section">
            <h2>Duration Analysis</h2>
        """
        duration_data = report['case_duration_analysis']
        if 'basic_statistics' in duration_data:
            html_content += "<h3>Basic Statistics (Hours)</h3>"
            for key, value in duration_data['basic_statistics'].items():
                html_content += f'<div class="metric"><strong>{key.title()}:</strong> {value:.2f}</div>'
        html_content += "</div>"
    
    # Activity Analysis
    if 'activity_analysis' in report and 'top_activities' in report['activity_analysis']:
        html_content += """
        <div class="section">
            <h2>Top Activities</h2>
            <table>
                <tr><th>Activity</th><th>Frequency</th></tr>
        """
        for activity, frequency in report['activity_analysis']['top_activities'].items():
            html_content += f"<tr><td>{activity}</td><td>{frequency:,}</td></tr>"
        html_content += "</table></div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return f"HTML report exported to {filename}"