import pandas as pd 
import numpy as np
from datetime import date 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
from pandas_datareader import data as pdr
import yfinance as yf
 
 # Create the date range for data
start_date = '2015-01-01'
end_date = date.today()
# Ticker symbol to investigate
ticker = 'SPY'
# Pull data for both low and high of day
df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
df['roll_high'] = df['High'].rolling(30, center=True).mean()
df['roll_low'] = df['Low'].rolling(30, center=True).mean()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['roll_high'], 'r-')
plt.plot(df.index, df['High'], 'k-')
plt.legend()
plt.tight_layout
plt.show();

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['roll_low'], 'r-')
plt.plot(df.index, df['Low'], 'k-')
plt.legend()
plt.tight_layout
plt.show();