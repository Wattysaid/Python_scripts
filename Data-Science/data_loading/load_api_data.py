"""
API Data Loader
---------------
Simple functions to fetch data from REST APIs.
"""

import requests
import pandas as pd
import json


def fetch_api_data(url, headers=None, params=None):
    """Fetch data from API endpoint."""
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


def api_to_dataframe(data):
    """Convert API response to DataFrame."""
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.json_normalize(data)
    else:
        return pd.DataFrame([data])