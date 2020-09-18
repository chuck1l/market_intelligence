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
yf.pdr_override()

class PredictPriceHigh(object):
    def __init__(self, dataset, window):
        self.window = window
        self.df = dataset
        self.df = self.df.astype('float32')

    def column_manipulation(self):
        self.df.columns = self.df.columns.str.lower()
        self.df.drop(['open', 'close', 'low', 'volume', 'adj close'], axis=1, inplace=True)
        self.df['tmr_high'] = self.df['high'].shift(periods=-1)
        self.df['tmr_high'].fillna(self.df['high'], inplace=True)

    def create_windows(self):
        # creating windows for the columns listed, window length parameter
        cols = [col for col in self.df.columns if col != 'tmr_high']  
        for i in range(self.window+1):
            for col in cols:
                self.df[col+'_back_'+str(i)] = self.df[col].shift(i, axis=0)
        self.df = self.df.dropna(axis=0, how='any')
        print(self.df.columns)
    
    def train_test_split(self):
        # Create the holdout sample for final prediction
        self.df, self.holdout_raw = self.df.drop(self.df.tail(1).index), self.df.tail(1)
        # Drop y from holdout
        self.holdout_raw.drop('tmr_high', axis=1, inplace=True)
        # Create the y set for high of day
        self.y = self.df['tmr_high']
        # Drop the y values from the dataframe
        self.X = self.df.drop('tmr_high', axis=1).copy()

    def reshape_tr_test(self):
        # Train and Test Split for High and Low of day models
        self.X_train_raw, self.X_test_raw, self.y_train_raw, self.y_test_raw = train_test_split(self.X, self.y, test_size=0.3, shuffle=True, random_state=42)
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
        model.add(LSTM(units=64))
        model.add(Dropout(0.25))
        # Output Layer
        model.add(Dense(units=1, activation='relu'))
        # Compiling the Neural Network
        model.compile(loss='mean_squared_error', optimizer ='adam')
        # Fit on training data
        model.fit(self.X_train, self.y_train, epochs=1000, batch_size=200, verbose=1)
        # Save Model 
        #model.save('../models/high_model.h5')
        # make predictions
        self.trainPredict = model.predict(self.X_train)
        self.testPredict = model.predict(self.X_test)
        # Calculate RMSE
        trainScore = math.sqrt(mean_squared_error(self.y_train, self.trainPredict))
        print('High of Day Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(self.y_test, self.testPredict))
        print('High of Day Test Score: %.2f RMSE' % (testScore))
    
    def get_predictions(self):
        self.column_manipulation()
        self.create_windows()
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
    window = 2
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    PredictPriceHigh(data, window).get_predictions()