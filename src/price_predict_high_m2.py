# Standard Imports
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
import math
from datetime import date 
from datetime import timedelta 
# Imports for LSTM Model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
# Source Data From Yahoo
from pandas_datareader import data as pdr
import yfinance as yf

def set_index_to_datetime(df):
    df.index = df['date']
    return None

class PredictPriceHigh(object):
    def __init__(self, dataset, days_back):
        self.days_back = days_back
        self.df = dataset
        self.df = self.df.astype('float32')

    def column_manipulation(self):
        self.df.reset_index(inplace=True)
        self.df.columns = self.df.columns.str.lower()
        set_index_to_datetime(self.df)
        adjust_close = self.df.pop('adj close')
        self.df.drop(['volume'], axis=1, inplace=True)
        self.df['avg_price'] = self.df.sum(axis=1)/self.df.shape[1]
        self.df.drop(['open'], axis=1, inplace=True)
        self.df['tmr_high'] = self.df['high'].shift(periods=-1)
        self.df['tmr_low'] = self.df['low'].shift(periods=-1)
        self.df['tmr_high'].fillna(self.df['high'], inplace=True)
        self.df['tmr_low'].fillna(self.df['low'], inplace=True)
        self.df['adjusted_close'] = adjust_close
        # Create a rolling mean
        self.df['roll_high'] = self.df['high'].rolling(self.days_back, center=True).mean()
        self.df['roll_low'] = self.df['low'].rolling(self.days_back, center=True).mean()
        self.df['roll_high'].fillna(self.df['high'], inplace=True)
        self.df['roll_low'].fillna(self.df['low'], inplace=True)
        self.df.drop('date', axis=1, inplace=True)
    
    def train_test_split(self):
        # Create the holdout sample for final prediction
        self.df, self.holdout_raw = self.df.drop(self.df.tail(100).index), self.df.tail(100)
        # Drop y from holdout and create holdout df for future plotting
        ho_cols = ['high', 'low', 'tmr_high', 'tmr_low']
        self.holdout_df = self.holdout_raw[ho_cols].copy()
        self.holdout_df.rename(columns={'low': 'persis_low', 'high': 'persis_high'}, inplace=True)
        # The true holdout dataframe is below
        self.holdout_raw.drop(['tmr_high', 'tmr_low', 'high', 'low', 'avg_price'], axis=1, inplace=True)
        # Create the y set for high and low of day
        self.y = self.df['tmr_high']
        # Drop the y values from the dataframe
        self.df.drop(['tmr_high', 'tmr_low', 'high', 'low', 'avg_price'], axis=1, inplace=True)
        # Train and Test Split for High and Low of day models
        train_end = int(self.df.shape[0]*.7)
        self.X_train_raw, self.X_test_raw = self.df.iloc[0:train_end, :].copy(), self.df.iloc[train_end:-1, :].copy()
        self.y_train_raw, self.y_test_raw = self.y.iloc[0:train_end].copy(), self.y.iloc[train_end:-1].copy()

    def reshape_tr_test(self):
        # Reshape the data for LSTM format
        self.X_train = np.expand_dims(self.X_train_raw.values[:, :], axis=2)
        self.X_test = np.expand_dims(self.X_test_raw.values[:, :], axis=2)
        self.holdout = np.expand_dims(self.holdout_raw, axis=2)
        self.y_train = self.y_train_raw.values.reshape(-1, 1)
        self.y_test = self.y_test_raw.values.reshape(-1, 1)
        
    def make_predictions(self):
        # Input Dimensions
        input_dim = self.X_train.shape[1]
        # Initializing the Neural Network Based On LSTM
        model = Sequential()
        # Adding 1st LSTM Layer
        model.add(LSTM(units=640, return_sequences=True, input_shape=(input_dim, 1)))
        model.add(Dropout(0.25))
        # Adding 2nd LSTM Layer
        model.add(LSTM(units=60))
        model.add(Dropout(0.25))
        # Output Layer
        model.add(Dense(units=1, activation='relu'))
        # Compiling the Neural Network
        model.compile(loss='mean_squared_error', optimizer ='adam')
        # Fit on training data
        model.fit(self.X_train, self.y_train, epochs=100, batch_size=25, verbose=1)
        # Save Model 
        #model.save('../models/high_model.h5')
        # make predictions
        self.trainPredict = model.predict(self.X_train)
        self.testPredict = model.predict(self.X_test)
        self.hoPredict = model.predict(self.holdout)
        # Calculate RMSE
        trainScore = math.sqrt(mean_squared_error(self.y_train, self.trainPredict))
        print('High of Day Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(self.y_test, self.testPredict))
        print('High of Day Test Score: %.2f RMSE' % (testScore))
        hoScore = math.sqrt(mean_squared_error(self.holdout_df['tmr_high'].values, self.hoPredict))
        print('High of Day Holdout Score: %.2f RMSE' % (hoScore))
    
    def get_predictions(self):
        self.column_manipulation()
        self.train_test_split()
        self.reshape_tr_test()
        self.make_predictions()
        # self.return_todays_pred()
        

if __name__ == '__main__':
     # Create the date range for data
    start_date = '2000-01-01'
    end_date = date.today()
    # Ticker symbol to investigate
    ticker = 'SPY'
    # Pull data for both low and high of day
    days_back = 20
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    PredictPriceHigh(data, days_back).get_predictions()