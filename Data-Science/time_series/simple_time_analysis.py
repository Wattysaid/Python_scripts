# Simple Time Series Analysis
import pandas as pd
import numpy as np
import sys

def moving_average(series, window=7):
    return series.rolling(window=window).mean()

def calculate_trend(series):
    x = np.arange(len(series))
    y = series.dropna().values
    if len(y) > 1:
        slope = np.polyfit(x[:len(y)], y, 1)[0]
        return slope
    return 0

def detect_seasonality(series, period=7):
    if len(series) < period * 2:
        return False
    
    seasonal_avg = []
    for i in range(period):
        values = series[i::period]
        seasonal_avg.append(values.mean())
    
    return np.std(seasonal_avg) / np.mean(seasonal_avg) > 0.1