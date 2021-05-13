import pandas as pd 
import numpy as np
from datetime import date 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#plt.style.use('ggplot')
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
df.drop(['Open', 'Close', 'Adj Close', 'Volume', 'Low'], axis=1, inplace=True)
df['after_training'] = df['High'].shift(periods=1)
df.dropna('rows', how='any', inplace=True)
df.reset_index(inplace=True)
df['Date'] = df['Date'].astype(str) 

indices = list(range(int(df.shape[0]*.7), int(df.shape[0])))
maximum = df['after_training'][0:int(df.shape[0]*.7)].max()
for i in indices:
    if df['after_training'][i] < maximum:
        df['after_training'][i] = df['after_training'][i]
    else:
        df['after_training'][i] = maximum

 #Plot the true vs persistence, high
xmarks=[i for i in range(1,len(df['Date'])+1,200)]
plt.figure(figsize=(18, 9))
plt.plot(df['Date'], df['after_training'], 'r-', label="Prediction")
plt.plot(df['Date'], df['High'], 'k--', label='True Value')
plt.axvline(indices[0], color='b', linestyle='-', label='X Train Ending')
plt.ylabel('Price ($)', fontsize=20, c='k')
plt.xlabel('Date', fontsize=20, c='k')
plt.xticks(xmarks, rotation=20)
plt.tick_params(axis='x', colors='k', labelsize=20)
plt.tick_params(axis='y', colors='k', labelsize=20)
plt.title('Feature and Target Engineering Understanding', fontsize=24)
plt.legend()
plt.tight_layout
plt.savefig('../imgs/feature_ing_explain.png')
plt.show();