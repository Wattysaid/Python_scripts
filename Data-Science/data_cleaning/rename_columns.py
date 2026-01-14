"""
Column Renamer
--------------
Simple script to rename columns in datasets.
"""

import pandas as pd
import sys

def ensure_packages():
    try:
        import pandas
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

def rename_columns(df):
    print("Current columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    df_renamed = df.copy()
    
    while True:
        choice = input("\nRename column? (column_number new_name) or 'done': ")
        if choice.lower() == 'done':
            break
        
        try:
            parts = choice.split(' ', 1)
            col_num = int(parts[0]) - 1
            new_name = parts[1]
            old_name = df.columns[col_num]
            df_renamed = df_renamed.rename(columns={old_name: new_name})
            print(f"Renamed '{old_name}' to '{new_name}'")
        except:
            print("Invalid input. Use: column_number new_name")
    
    return df_renamed