"""
Variant Analysis
---------------
Simple functions to analyze process variants in event logs.
"""

import pandas as pd
from collections import Counter


def extract_variants(df):
    """Extract process variants from event log."""
    if 'timestamp' in df.columns:
        df = df.sort_values(['case_id', 'timestamp'])
    
    variants = {}
    variant_cases = {}
    
    for case_id, group in df.groupby('case_id'):
        trace = tuple(group['activity'].tolist())
        
        if trace not in variants:
            variants[trace] = 0
            variant_cases[trace] = []
        
        variants[trace] += 1
        variant_cases[trace].append(case_id)
    
    return variants, variant_cases


def calculate_variant_statistics(variants, total_cases):
    """Calculate statistics for each variant."""
    variant_stats = []
    
    for i, (variant, frequency) in enumerate(sorted(variants.items(), key=lambda x: x[1], reverse=True), 1):
        stats = {
            'variant_id': i,
            'variant': ' -> '.join(variant),
            'frequency': frequency,
            'percentage': (frequency / total_cases) * 100,
            'trace_length': len(variant),
            'unique_activities': len(set(variant))
        }
        variant_stats.append(stats)
    
    return pd.DataFrame(variant_stats)


def find_common_patterns(variants, min_length=2):
    """Find common sub-patterns in variants."""
    patterns = Counter()
    
    for variant, frequency in variants.items():
        # Extract all sub-sequences of minimum length
        for i in range(len(variant) - min_length + 1):
            for j in range(i + min_length, len(variant) + 1):
                pattern = variant[i:j]
                if len(pattern) >= min_length:
                    patterns[pattern] += frequency
    
    return patterns