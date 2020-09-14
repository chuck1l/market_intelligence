import pandas as pd  
import numpy as np
from price_predict_high import PredictPriceHigh
from price_predict_low import PredictPriceLow
from datetime import date 
from datetime import timedelta 
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

if __name__ == '__main__':
    # Create the date range for data
    start_date = '2000-01-01'
    end_date = date.today()
    # Ticker symbol to investigate
    print('Hello, please enter the ticker symbol of interest (all caps).')
    ticker = input()
    # Pull data for both low and high of day
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    # Create df for high, and low price
    data_high = data['High']
    data_low = data['Low']
    # Train models for High, Low
    PredictPriceHigh(data_high).get_predictions()
    PredictPriceLow(data_low).get_predictions()



    
    
    




    
