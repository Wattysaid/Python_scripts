"""
Conformance Checker
-------------------
Simple conformance checking between event log and process model.
"""

import pandas as pd
import sys

def ensure_packages():
    try:
        import pandas
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

def extract_traces(df):
    if 'timestamp' in df.columns:
        df = df.sort_values(['case_id', 'timestamp'])
    return df.groupby('case_id')['activity'].apply(tuple).to_dict()

def check_conformance(traces, allowed_relations):
    conformant_cases = []
    non_conformant_cases = []
    
    for case_id, trace in traces.items():
        is_conformant = True
        violations = []
        
        for i in range(len(trace) - 1):
            current = trace[i]
            next_act = trace[i + 1]
            
            if (current, next_act) not in allowed_relations:
                is_conformant = False
                violations.append(f"{current} -> {next_act}")
        
        if is_conformant:
            conformant_cases.append(case_id)
        else:
            non_conformant_cases.append({
                'case_id': case_id,
                'violations': violations,
                'trace': ' -> '.join(trace)
            })
    
    return conformant_cases, non_conformant_cases