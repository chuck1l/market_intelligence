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

class PredictPriceLow(object):
    def __init__(self, dataset, window):
        self.window = window
        self.df = dataset
        self.df = self.df.astype('float32')

    def column_manipulation(self):
        self.df.columns = self.df.columns.str.lower()
        self.df['tmr_low'] = self.df['low'].shift(periods=-1)
        self.df['tmr_low'].fillna(self.df['low'], inplace=True)

    def create_windows(self):
        # creating windows for the columns listed, window length parameter
        cols = [col for col in self.df.columns if col != 'tmr_low']  
        for i in range(self.window+1):
            for col in cols:
                self.df[col+'_back_'+str(i)] = self.df[col].shift(i, axis=0)
        self.df = self.df.dropna(axis=0, how='any')
    
    def train_test_split(self):
        # Create the holdout sample for final prediction
        self.df, self.holdout_raw = self.df.drop(self.df.tail(1).index), self.df.tail(1)
        # Drop y from holdout
        self.holdout_raw.drop('tmr_low', axis=1, inplace=True)
        # Create the y set for Low of Day
        self.y = self.df['tmr_low']
        # Drop the y values from the original dataframe
        self.X = self.df.drop('tmr_low', axis=1).copy()

    def reshape_tr_test(self):
        # Train and Test split for Low of Day models
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
        # Adding 2nd LSTM Layer
        model.add(LSTM(units=64))
        # Adding Dropout
        model.add(Dropout(0.25))
        # Output Layer
        model.add(Dense(units=1, activation='relu'))
        # Compiling the Neural Network
        model.compile(loss='mean_squared_error', optimizer ='adam')
        # Fit on training data
        model.fit(self.X_train, self.y_train, epochs=300, batch_size=1024, verbose=1)
        # Save Model 
        #model.save('../models/high_model.h5')
        # make predictions
        self.trainPredict = model.predict(self.X_train)
        self.testPredict = model.predict(self.X_test)
        # Calculate RMSE
        trainScore = math.sqrt(mean_squared_error(self.y_train, self.trainPredict))
        print('Low of Day Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(self.y_test, self.testPredict))
        print('Low of Day Test Score: %.2f RMSE' % (testScore))

    # def get_plots(self):
    #     # Plot The Train Results
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(self.train_x[self.look_back+1:], self.trainPredict, 'r--', label='Predicted')
    #     plt.plot(self.train_x[self.look_back+1:], self.y_train.reshape(-1, 1), 'k--', label='True Value')
    #     plt.ylabel('Price ($)', fontsize=16)
    #     plt.xlabel('Historical Dates', fontsize=16)
    #     plt.xticks([])
    #     plt.title('Train Data Predicted and True Tomorrow High Price', fontsize=19)
    #     plt.legend()
    #     plt.tight_layout
    #     #plt.savefig('../imgs/high_train_lstm_tuned.png')
    #     plt.show();

    #     # Plot the results
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(self.test_x[self.look_back+1:], self.testPredict, 'r--', label='Predicted')
    #     plt.plot(self.test_x[self.look_back+1:], self.y_test.reshape(-1, 1), 'k--', label='True Value')
    #     plt.ylabel('Price ($)', fontsize=16)
    #     plt.xlabel('Historical Dates', fontsize=16)
    #     plt.xticks([])
    #     plt.title('Test Data Predicted and True Tomorrow High Price', fontsize=19)
    #     plt.legend()
    #     plt.tight_layout
    #     #plt.savefig('../imgs/high_test_lstm_tuned.png')
    #     plt.show();
    
    # def return_todays_pred(self):
    #     finalX = np.array([[self.holdout[0], self.holdout[1]]])
    #     finalX = np.reshape(finalX, (finalX.shape[0], 1, finalX.shape[1]))
    #     v = load_model('../models/high_model.h5')
    #     finalPredict = v.predict(finalX)
    #     finalPredict = self.scaler.inverse_transform(finalPredict)
    #     print(f"I am tomorrow's high: {finalPredict[0][0]:.2f}")
        
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
    PredictPriceLow(data, window).get_predictions()