"""
CSV File Loader
---------------
Simple functions to load CSV files with various options.
"""

import pandas as pd


def load_csv(file_path, delimiter=',', encoding='utf-8', header=0, 
             skip_rows=0, nrows=None, usecols=None):
    """
    Load a CSV file with configurable options.
    
    Parameters:
    -----------
    file_path : str - Path to the CSV file
    delimiter : str - Column delimiter (default: ',')
    encoding : str - File encoding (default: 'utf-8')
    header : int or None - Row number for header (default: 0)
    skip_rows : int - Number of rows to skip at the start
    nrows : int or None - Number of rows to read
    usecols : list or None - Columns to read
    
    Returns:
    --------
    pandas.DataFrame
    """
    df = pd.read_csv(
        file_path,
        delimiter=delimiter,
        encoding=encoding,
        header=header,
        skiprows=skip_rows,
        nrows=nrows,
        usecols=usecols
    )
    return df


def preview_data(df, rows=5):
    """Display preview of the dataframe."""
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nFirst {rows} rows:")
    print(df.head(rows))
    print(f"\nColumn Types:")
    print(df.dtypes)