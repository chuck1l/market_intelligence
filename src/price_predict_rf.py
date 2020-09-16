# Standard Imports
import pandas as pd
import numpy as np
from datetime import date 
from datetime import timedelta 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
import seaborn as sns
# All Imports For The Models
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
# Source Data From Yahoo
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

class RandomForestPredict(object):
    def __init__(self, data, window):
        self.df = data
        self.window = window

    def column_manipulation(self):
        self.df.columns = self.df.columns.str.lower()
        cols = self.df.columns
        if 'adj close' in cols:
            self.df.drop('adj close', axis=1, inplace=True)
        self.df['tmr_high'] = self.df['high'].shift(periods=-1)
        self.df['tmr_low'] = self.df['low'].shift(periods=-1)
        self.df['tmr_high'].fillna(self.df['high'], inplace=True)
        self.df['tmr_low'].fillna(self.df['low'], inplace=True)
    
    def create_windows(self):
        # creating windows for the columns listed, window length parameter
        cols = ['open', 'high', 'low', 'close', 'volume']
        for i in range(self.window+1):
            for col in cols:
                data[col+'_back_'+str(i)] = data[col].shift(i, axis=0)
        self.df = self.df.dropna(axis=0, how='any')

    def train_test_holdout(self):
        # Create the train, test and holdout dataframes
        train_len = int(self.df.shape[0] * .75)
        self.train = self.df.iloc[0:train_len, :].copy()
        self.test = self.df.iloc[train_len:-1, :].copy()
        self.holdout = self.df.iloc[-1:, :].copy()
        # Create the true y sets for train, test
        self.y_high_train = self.train['tmr_high'].values
        self.y_low_train = self.train['tmr_low'].values
        self.y_high_test = self.test['tmr_high'].values
        self.y_low_test = self.test['tmr_low'].values
        # Create the X train, test
        self.X_train = self.train.drop(['tmr_high', 'tmr_low'], axis=1).values
        self.X_test = self.test.drop(['tmr_high', 'tmr_low'], axis=1).values
        # Drop tomorrow high and tomorrow low for holdout
        self.holdout.drop(['tmr_high', 'tmr_low'], axis=1, inplace=True)
        self.holdout = self.holdout.values
        # Reshape the y datasets
        self.y_high_train = self.y_high_train.reshape(-1, 1)
        self.y_low_train = self.y_low_train.reshape(-1, 1)
        self.y_high_test = self.y_high_test.reshape(-1, 1)
        self.y_low_test = self.y_low_test.reshape(-1, 1)
    
    def scale_data(self):
        self.X_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.y_high_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.y_low_scaler = MinMaxScaler(feature_range=(-1, 1))
        # Scale the X datasets (and holdout)
        self.sc_X_train = self.X_scaler.fit_transform(self.X_train)
        self.sc_X_test = self.X_scaler.transform(self.X_test)
        self.sc_holdout = self.X_scaler.transform(self.holdout)
        # Scale the Y datasets
        self.sc_y_high_train = self.y_high_scaler.fit_transform(self.y_high_train)
        self.sc_y_high_test = self.y_high_scaler.transform(self.y_high_test)
        self.sc_y_low_train = self.y_low_scaler.fit_transform(self.y_low_train)
        self.sc_y_low_test = self.y_low_scaler.transform(self.y_low_test)

    def parameter_selection_high(self):
        # Parameters to investigate
        num_estimators = [600, 800, 900, 1000]
        max_features = [4, 6, 8]
        max_depth = [10, 15, 20, 25]
        min_samples_split = [2, 3, 4]
        min_samples_leaf = [1, 2, 3]
        # build a dictionary containing the parameters
        param_grid = {'n_estimators': num_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
        # Create the model, Random Forest
        regressor = RandomForestRegressor()
        # Iterations to run
        desired_iterations = 100
        # Build the randomized search cross validation
        random_search = RandomizedSearchCV(regressor, 
                                   param_grid, 
                                   scoring='neg_root_mean_squared_error',
                                   cv=3,
                                   n_iter=desired_iterations,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1)
        random_search.fit(self.sc_X_train, np.ravel(self.sc_y_high_train))
        randomforest_randomsearch_bestparams = random_search.best_params_
        print('Random Forest Params', random_search.best_params_)  
        self.randomforest_randomsearch_bestscore = -1 * random_search.best_score_
        print(f'Random Forest Score: {self.randomforest_randomsearch_bestscore:0.3f}')  # negative root mean square error

    def parameter_selection_low(self):
        # Parameters to investigate
        num_estimators = [600, 800, 900, 1000]
        max_features = [4, 6, 8]
        max_depth = [10, 15, 20, 25]
        min_samples_split = [2, 3, 4]
        min_samples_leaf = [1, 2, 3]
        # build a dictionary containing the parameters
        param_grid = {'n_estimators': num_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
        # Create the model, Random Forest
        regressor = RandomForestRegressor()
        # Iterations to run
        desired_iterations = 100
        # Build the randomized search cross validation
        random_search = RandomizedSearchCV(regressor, 
                                   param_grid, 
                                   scoring='neg_root_mean_squared_error',
                                   cv=3,
                                   n_iter=desired_iterations,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1)
        random_search.fit(self.sc_X_train, np.ravel(self.sc_y_low_train))
        randomforest_randomsearch_bestparams = random_search.best_params_
        print('Random Forest Params', random_search.best_params_)  
        self.randomforest_randomsearch_bestscore = -1 * random_search.best_score_
        print(f'Random Forest Score: {self.randomforest_randomsearch_bestscore:0.3f}')  # negative root mean square error

    def get_predictions(self):
        self.column_manipulation()
        self.create_windows()
        self.train_test_holdout()
        self.scale_data()
        self.parameter_selection_high()

if __name__ == '__main__':
     # Create the date range for data
    start_date = '2000-01-01'
    end_date = date.today()
    # Ticker symbol to investigate
    ticker = 'SPY'
    # Pull data for both low and high of day
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    RandomForestPredict(data, 2).get_predictions()