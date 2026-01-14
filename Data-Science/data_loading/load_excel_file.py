"""
Excel File Loader
-----------------
Simple functions to load Excel files with multiple sheets support.
"""

import pandas as pd


def list_sheets(file_path):
    """List all sheets in an Excel file."""
    xl = pd.ExcelFile(file_path)
    return xl.sheet_names


def load_excel(file_path, sheet_name=0, header=0, skip_rows=0, 
               usecols=None, nrows=None):
    """
    Load an Excel file with configurable options.
    
    Parameters:
    -----------
    file_path : str - Path to the Excel file
    sheet_name : str/int/list - Sheet(s) to load
    header : int or None - Row number for header
    skip_rows : int - Number of rows to skip
    usecols : str or list - Columns to read (e.g., "A:C" or [0,1,2])
    nrows : int or None - Number of rows to read
    
    Returns:
    --------
    pandas.DataFrame or dict of DataFrames
    """
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=header,
        skiprows=skip_rows,
        usecols=usecols,
        nrows=nrows
    )
    return df


def load_all_sheets(file_path):
    """Load all sheets from an Excel file into a dictionary."""
    return pd.read_excel(file_path, sheet_name=None)


def preview_sheet(df, sheet_name="Sheet"):
    """Display preview of a sheet."""
    print(f"\n{'='*60}")
    print(f"SHEET: {sheet_name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData Types:")
    print(df.dtypes)