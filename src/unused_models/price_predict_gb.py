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

class GradientBoostPredict(object):
    def __init__(self, data, window):
        self.df = data
        self.window = window

    def column_manipulation(self):
        self.df.columns = self.df.columns.str.lower()
        self.df.drop(['volume', 'open', 'close'], axis=1, inplace=True)
        cols = self.df.columns
        if 'adj close' in cols:
            self.df.drop('adj close', axis=1, inplace=True)
        self.df['tmr_high'] = self.df['high'].shift(periods=-1)
        self.df['tmr_low'] = self.df['low'].shift(periods=-1)
        self.df['tmr_high'].fillna(self.df['high'], inplace=True)
        self.df['tmr_low'].fillna(self.df['low'], inplace=True)
    
    def create_windows(self):
        # creating windows for the columns listed, window length parameter
        cols = ['high', 'low']
        for i in range(self.window+1):
            for col in cols:
                data[col+'_back_'+str(i)] = data[col].shift(i, axis=0)
        self.df = self.df.dropna(axis=0, how='any')

    def train_test_holdout(self):
        # Create the train, test and holdout dataframes
        train_len = int(self.df.shape[0] * .67)
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
        
    def scale_data(self):
        # Reshape the y datasets
        self.y_high_train = self.y_high_train.reshape(-1, 1)
        self.y_low_train = self.y_low_train.reshape(-1, 1)
        self.y_high_test = self.y_high_test.reshape(-1, 1)
        self.y_low_test = self.y_low_test.reshape(-1, 1)
        # Create the scalers
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
        num_estimators = [200, 250, 300, 400]
        max_features = [6, 8, 10, 15]
        max_depth = [13, 14, 15, 16]
        min_samples_split = [2, 3, 4]
        min_samples_leaf = [1, 2, 3]
        # build a dictionary containing the parameters
        param_grid = {'n_estimators': num_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
        # Create the model, Random Forest
        high_regressor = RandomForestRegressor()
        # Iterations to run
        desired_iterations = 100
        # Build the randomized search cross validation
        high_random_search = RandomizedSearchCV(high_regressor, 
                                   param_grid, 
                                   scoring='neg_root_mean_squared_error',
                                   cv=5,
                                   n_iter=desired_iterations,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1)
        high_random_search.fit(self.X_train, self.y_high_train)
        randomforest_high_bestparams = high_random_search.best_params_
        print('Random Forest High of Day Training Params', high_random_search.best_params_)  
        randomforest_high_bestscore = -1 * high_random_search.best_score_
        print(f'Random Forest High of Day Training Score: {randomforest_high_bestscore:0.3f}')  # negative root mean square error
        # Assign the best parameters, High of Day Model
        self.high_n_estimators = randomforest_high_bestparams['n_estimators']
        self.high_min_samples_split = randomforest_high_bestparams['min_samples_split']
        self.high_min_saplies_leaf = randomforest_high_bestparams['min_samples_leaf']
        self.high_max_features = randomforest_high_bestparams['max_features']
        self.high_max_depth = randomforest_high_bestparams['max_depth']

    def parameter_selection_low(self):
        # Parameters to investigate
        num_estimators = [450, 500, 550, 600,]
        max_features = [3, 4, 5]
        max_depth = [19, 20, 22, 23]
        min_samples_split = [2, 3, 4]
        min_samples_leaf = [2, 3, 4]
        # build a dictionary containing the parameters
        param_grid = {'n_estimators': num_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
        # Create the model, Random Forest
        low_regressor = RandomForestRegressor()
        # Iterations to run
        desired_iterations = 100
        # Build the randomized search cross validation
        low_random_search = RandomizedSearchCV(low_regressor, 
                                   param_grid, 
                                   scoring='neg_root_mean_squared_error',
                                   cv=5,
                                   n_iter=desired_iterations,
                                   verbose=1,
                                   return_train_score=True,
                                   n_jobs=-1)
        low_random_search.fit(self.sc_X_train, np.ravel(self.sc_y_low_train))
        randomforest_low_bestparams = low_random_search.best_params_
        print('Random Forest Low of Day Training Params', low_random_search.best_params_)  
        randomforest_low_bestscore = -1 * low_random_search.best_score_
        print(f'Random Forest Low of Day Training Score: {randomforest_low_bestscore:0.3f}')  # negative root mean square error
        # Assign the best parameters, Low of Day Model
        self.low_n_estimators = randomforest_low_bestparams['n_estimators']
        self.low_min_samples_split = randomforest_low_bestparams['min_samples_split']
        self.low_min_saplies_leaf = randomforest_low_bestparams['min_samples_leaf']
        self.low_max_features = randomforest_low_bestparams['max_features']
        self.low_max_depth = randomforest_low_bestparams['max_depth']

    def run_model_high(self):
        # Random Forest Model For High of Day
        high_model = RandomForestRegressor(max_features=self.high_max_features,
                                            max_depth=self.high_max_depth,
                                            min_samples_leaf=self.high_min_saplies_leaf,
                                            min_samples_split=self.high_min_samples_split,
                                            n_estimators=self.high_n_estimators)
        high_model.fit(self.X_train, self.y_high_train)
        y_hat_high = high_model.predict(self.X_test)
        # Inverse the scaling
        #y_hat_high = self.y_high_scaler.inverse_transform(y_hat_high.reshape(-1, 1))
        # RMSE on non-scaled values
        rmse_high = np.sqrt(mean_squared_error(y_hat_high, self.y_high_test))
        # Print the test results for High of Day
        print(f'Random Forest RMSE High of Day: {rmse_high:0.3f}')

    def run_model_low(self):
        # Random Forest Model For High of Day
        low_model = RandomForestRegressor(max_features=self.low_max_features,
                                            max_depth=self.low_max_depth,
                                            min_samples_leaf=self.low_min_saplies_leaf,
                                            min_samples_split=self.low_min_samples_split,
                                            n_estimators=self.low_n_estimators)
        low_model.fit(self.sc_X_train, np.ravel(self.sc_y_low_train))
        y_hat_low = low_model.predict(self.sc_X_test)
        # Inverse the scaling
        y_hat_low = self.y_low_scaler.inverse_transform(y_hat_low.reshape(-1, 1))
        # RMSE on non-scaled values
        rmse_low = np.sqrt(mean_squared_error(y_hat_low, self.y_low_test))
        # Print the test results for High of Day
        print(f'Random Forest RMSE High of Day: {rmse_low:0.3f}')

    def get_predictions(self):
        self.column_manipulation()
        self.create_windows()
        self.train_test_holdout()
        #self.scale_data()
        self.parameter_selection_high()
        self.run_model_high()
        # self.parameter_selection_low()
        # self.run_model_low()

if __name__ == '__main__':
     # Create the date range for data
    start_date = '2000-01-01'
    end_date = date.today()
    # Ticker symbol to investigate
    ticker = 'SPY'
    # Pull data for both low and high of day
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    RandomForestPredict(data, 10).get_predictions()