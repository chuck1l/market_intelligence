import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from datetime import datetime 

class DataClean(object):
    '''
    The data should be delivered as csv, stock market security of interest
    in the timeframe of 1 hour in this analysis. All indicators will be considered 
    and filtered to principal components that provide value to the signal.
    '''
    def __init__(self, location):
        self.data = pd.read_csv(location)
    def date_time(self):
        self.data['index'] = pd.to_datetime(self.data['time'], unit='s') - pd.Timedelta(hours=7)
        self.data['date'] = self.data['index'].dt.date
        self.data['time_of_day'] = self.data['index'].dt.hour
        self.data.set_index(self.data['index'], inplace=True)
        self.data.drop(['time', 'index'], axis=1, inplace=True)
    def create_day_high_low_col(self):
        mask = (self.data['time_of_day'] >= 7) & (self.data['time_of_day'] <= 14)
        # Only considering market hours of 7AM to 2PM Mountain standard time (ignore pre/post)
        self.data['high_of_day'] = self.data[mask].groupby(['date'])['high'].transform(max)
        self.data['low_of_day'] = self.data[mask].groupby(['date'])['low'].transform(min)
        # Fill the pre and post market hours with high/low from normal market hours
        self.data['high_of_day'] = self.data.groupby(['date'])['high_of_day'].transform(max)
        self.data['low_of_day'] = self.data.groupby(['date'])['low_of_day'].transform(min)
        self.data['hi_lo_delta'] = round((self.data['high_of_day'] - self.data['low_of_day']) / self.data['low_of_day'] * 100, 2)
        self.data['average_price'] = round((self.data['high_of_day'] + self.data['low_of_day'])/2, 2)
    def time_at_high(self):
        # Again, still only interested in regular market hours
        mask = (self.data['time_of_day'] >= 7) & (self.data['time_of_day'] <= 14)
        # Identify the index of the high of day, then one hot incode
        hi_index = self.data[mask].groupby(['date'])['high'].idxmax()
        self.data['time_at_high'] = 0
        for i in hi_index:
            self.data.loc[[i], 'time_at_high'] = 1
    def time_at_low(self):
        # Again, still only interested in regular market hours
        mask = (self.data['time_of_day'] >= 7) & (self.data['time_of_day'] <= 14)
        # Identify the index of the high of day, then one hot incode
        hi_index = self.data[mask].groupby(['date'])['low'].idxmin()
        self.data['time_at_low'] = 0
        for i in hi_index:
            self.data.loc[[i], 'time_at_low'] = 1
    def create_tomorrow_cols(self):
        # I need to predict tomorrow's values, so shifting them to the
        # today's information to prevent data leakage
        self.data['tomorrow_high_of_day'] = self.data['high_of_day'].shift(-16)
        self.data['tomorrow_low_of_day'] = self.data['low_of_day'].shift(-16)
        self.data['tomorrow_time_at_high'] = self.data['time_at_high'].shift(-16)
        self.data['tomorrow_time_at_low'] = self.data['time_at_low'].shift(-16)
    def drop_all_nan_cols(self):
        threshold = self.data.shape[0] - 500
        self.data.dropna(axis='columns', thresh=threshold, inplace=True)
    def drop_any_nan_rows(self):
        self.data.dropna(axis='rows', how='any', inplace=True)
    # Run all function in this class together, sequential order   
    def data_prepared(self):
        self.date_time()
        self.create_day_high_low_col()
        self.time_at_high()
        self.time_at_low()
        self.create_tomorrow_cols()
        self.drop_all_nan_cols()
        self.drop_any_nan_rows()
        return self.data
        
if __name__ == '__main__':
    location = '../data/spy_1h_ext_hours.csv'
    spy = DataClean('../data/spy_1h_ext_hours.csv')
    spy.data_prepared()

    