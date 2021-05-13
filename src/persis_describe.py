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
df.drop(['Open', 'Close', 'Adj Close', 'Volume'], axis=1, inplace=True)
df.reset_index(inplace=True)
df['Date'] = df['Date'].astype(str)
# Create tomorrow's prediction
df['pred_high'] = df['High'].shift(periods=1)
df['pred_low'] = df['Low'].shift(periods=1)
# Create tomorrow's actual
df['tmr_high'] = df['High'].shift(periods=-1)
df['tmr_low'] = df['Low'].shift(periods=-1)
df.dropna(axis='rows', how='any', inplace=True)
df2 = df.iloc[-100:, :]
print(df2.head())

rmse_pm_high = np.sqrt(mean_squared_error(df2['pred_high'], df2['High']))
rmse_pm_low = np.sqrt(mean_squared_error(df2['pred_low'], df2['Low']))
# Print the test results for High of Day
print(f'Baseline RMSE High of Day: {rmse_pm_high:0.3f}')
print(f'Baseline RMSE Low of Day: {rmse_pm_low:0.3f}')


data_comb = pd.read_csv('../data/rf_graphing_df.csv')

toggle = True
if toggle:
    #Plot the true vs persistence, high
    xmarks=[i for i in range(1,len(df2['Date'])+1,19)]
    plt.figure(figsize=(18, 9))
    plt.plot(df2['Date'], df2['pred_high'], 'r-', label="Prediction")
    plt.plot(df2['Date'], df2['High'], 'k--', label='True Value')
    plt.ylabel('Price ($)', fontsize=20, c='k')
    plt.xlabel('Date', fontsize=20, c='k')
    plt.xticks(xmarks, rotation=0)
    plt.tick_params(axis='x', colors='k', labelsize=20)
    plt.tick_params(axis='y', colors='k', labelsize=20)
    plt.title('Persistence Model Prediction vs True Value, High', fontsize=24)
    plt.legend()
    plt.tight_layout
    plt.savefig('../imgs/pm_high_final_graph.png')
    plt.show();
