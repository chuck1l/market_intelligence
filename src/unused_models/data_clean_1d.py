import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from datetime import datetime 

class DataCleanOneDay(object):
    '''
    The data should be delivered as csv, stock market security of interest
    in the timeframe of 1 day in this analysis. All indicators will be considered 
    and filtered to principal components that provide value to the signal.
    '''
    def __init__(self, location):
        self.data = pd.read_csv(location)

    def date_time(self):
        self.data['index'] = pd.to_datetime(self.data['time'], unit='s') #- pd.Timedelta(hours=7)
        self.data['date'] = self.data['index'].dt.date
        self.data['year'] = self.data['index'].dt.year
        self.data['day_of_week'] = self.data['index'].dt.dayofweek
        self.data['day_of_year'] = self.data['index'].dt.dayofyear
        self.data['month'] = self.data['index'].dt.month
        self.data.set_index(self.data['index'], inplace=True)
        self.data.drop(['time', 'index'], axis=1, inplace=True)

    def create_new_price_cols(self):
        self.data['hi_lo_delta'] = round((self.data['high'] - self.data['low']) / self.data['low'] * 100, 2)
        self.data['average_price'] = round((self.data['high'] + self.data['low'])/2, 2)

    def create_tomorrow_cols(self):
        # I need to predict tomorrow's values, so shifting them to the
        # today's information to prevent data leakage
        self.data['tomorrow_high'] = self.data['high'].shift(-1)
        self.data['tomorrow_low'] = self.data['low'].shift(-1)

    def drop_all_nan_cols(self):
        threshold = self.data.shape[0] - 500
        self.data.dropna(axis='columns', thresh=threshold, inplace=True)

    def drop_any_nan_rows(self):
        self.data.dropna(axis='rows', how='any', inplace=True)
    # Run all function in this class together, sequential order 
      
    def data_prepared(self):
        self.date_time()
        self.create_new_price_cols()
        self.create_tomorrow_cols()
        self.drop_all_nan_cols()
        self.drop_any_nan_rows()
        mask = self.data['year'] >= 2000
        self.data = self.data[mask] 
        self.data['ticks'] = range(0, len(self.data.index.values))
        return self.data
        
if __name__ == '__main__':
    location = '../data/spy_1d.csv'
    spy = DataCleanOneDay(location)
    spy.data_prepared()