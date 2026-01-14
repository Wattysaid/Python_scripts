"""
Database Connection Loader
--------------------------
Simple functions to connect to databases and load data.
"""

import pandas as pd
from sqlalchemy import create_engine


def load_from_database(connection_string, query, params=None):
    """Load data from database using SQL query."""
    engine = create_engine(connection_string)
    df = pd.read_sql(query, engine, params=params)
    return df