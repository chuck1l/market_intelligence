# Standard Imports
import pandas as pd
import numpy as np
import pickle
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
        self.df.drop(['volume', 'open'], axis=1, inplace=True)
        self.df['tmr_high'] = self.df['high'].shift(periods=-1)
        self.df['tmr_low'] = self.df['low'].shift(periods=-1)
        self.df['tmr_high'].fillna(self.df['high'], inplace=True)
        self.df['tmr_low'].fillna(self.df['low'], inplace=True)
    
    def create_windows(self):
        # creating windows for the columns listed, window length parameter
        cols = ['high', 'low', 'adj close', 'close']
        for i in range(self.window+1):
            for col in cols:
                data[col+'_back_'+str(i)] = data[col].shift(i, axis=0)
        self.df = self.df.dropna(axis=0, how='any')

    def train_test_holdout(self):
        # Create the holdout sample for final prediction
        self.df_2, self.holdout = self.df.drop(self.df.tail(1).index), self.df.tail(1)
        # Drop y from holdout
        self.holdout.drop(['tmr_high', 'tmr_low'], axis=1, inplace=True)
        # Create the y set for high and low of day
        self.y_h = self.df_2['tmr_high']
        self.y_l = self.df_2['tmr_low']
        # Drop the y values from the dataframe
        self.df_2.drop(['tmr_high', 'tmr_low'], axis=1, inplace=True)
        # Train and Test Split for High and Low of day models
        self.X_train_h, self.X_test_h, self.y_train_h, self.y_test_h = train_test_split(self.df_2, self.y_h, test_size=0.3, shuffle=True, random_state=42)
        self.X_train_l, self.X_test_l, self.y_train_l, self.y_test_l = train_test_split(self.df_2, self.y_l, test_size=0.3, shuffle=True, random_state=42)
        # Extract High Values only
        self.X_train_h_v, self.X_test_h_v = self.X_train_h.copy().values, self.X_test_h.copy().values
        self.y_train_h_v, self.y_test_h_v = self.y_train_h.copy().values, self.y_test_h.copy().values
        # Extract Low values only
        self.X_train_l_v, self.X_test_l_v = self.X_train_l.copy().values, self.X_test_l.copy().values
        self.y_train_l_v, self.y_test_l_v = self.y_train_l.copy().values, self.y_test_l.copy().values

    def parameter_selection_high(self):
        # Parameters to investigate
        num_estimators = [200, 250, 300, 350, 400, 450]
        max_features = [10, 15, 17, 19, 20]
        max_depth = [13, 14, 15, 16]
        min_samples_split = [2, 3, 4, 6]
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
        high_random_search.fit(self.X_train_h_v, self.y_train_h_v)
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
        num_estimators = [100, 150, 200, 250, 300]
        max_features = [22, 26, 28, 30]
        max_depth = [10, 11, 12, 13, 14]
        min_samples_split = [2, 3, 4, 6]
        min_samples_leaf = [1, 2, 3]
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
        low_random_search.fit(self.X_train_l_v, np.ravel(self.y_train_l_v))
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
        high_model.fit(self.X_train_h_v, self.y_train_h_v)
        # Pickle the model
        high_filename = ('high_rfmodel.sav')
        pickle.dump(high_model, open(high_filename, 'wb'))
        # Predict high on test
        self.y_hat_high = high_model.predict(self.X_test_h_v)
        # Calculate RMSE
        rmse_high = np.sqrt(mean_squared_error(self.y_hat_high, self.y_test_h_v))
        # Print the test results for High of Day
        print(f'Random Forest RMSE High of Day: {rmse_high:0.3f}')

    def run_model_low(self):
        # Random Forest Model For High of Day
        low_model = RandomForestRegressor(max_features=self.low_max_features,
                                            max_depth=self.low_max_depth,
                                            min_samples_leaf=self.low_min_saplies_leaf,
                                            min_samples_split=self.low_min_samples_split,
                                            n_estimators=self.low_n_estimators)
        low_model.fit(self.X_train_l_v, np.ravel(self.y_train_l_v))
        # Pickle the model
        low_filename = ('low_rfmodel.sav')
        pickle.dump(low_model, open(low_filename, 'wb'))
        # Predict low on test
        self.y_hat_low = low_model.predict(self.X_test_l_v)
        # Calculate RMSE 
        rmse_low = np.sqrt(mean_squared_error(self.y_hat_low, self.y_test_l_v))
        # Print the test results for High of Day
        print(f'Random Forest RMSE High of Day: {rmse_low:0.3f}')

    def create_result_dataframe(self):
        # Create Results DataFrame
        self.result_df = pd.DataFrame(data=self.y_test_h)
        print(self.result_df.shape)
        self.result_df = pd.concat([self.result_df, self.y_test_l], axis=1)
        print(self.result_df.shape)
        print(self.result_df.head())
        self.result_df['y_hat_high'] = self.y_hat_high
        self.result_df['y_hat_low'] = self.y_hat_low
        self.result_df.sort_index(inplace=True)

    def todays_predictions(self):
        high_filename = ('high_rfmodel.sav')
        high_model = pickle.load(open(high_filename, 'rb'))
        low_filename = ('low_rfmodel.sav')
        low_model = pickle.load(open(low_filename, 'rb'))
        # Predict and instantiate today's results
        self.high_result = high_model.predict(self.holdout)
        self.low_result = low_model.predict(self.holdout)
        print(self.high_result, self.low_result)

    def get_plots_high(self):
        # Plot The Train Results
        plt.figure(figsize=(12, 6))
        plt.plot(self.result_df.index, self.result_df['y_hat_high'], c='r', marker='o', markersize=1, label='Prediction')
        plt.plot(self.result_df.index, self.result_df['tmr_high'], 'k--', label='True Value')
        plt.ylabel('Price ($)', fontsize=16, c='k')
        plt.xlabel('Daily Samples Back to January 2000', fontsize=16, c='k')
        plt.xticks([])
        plt.title('Predictions and True High of Day Price', fontsize=19)
        plt.legend()
        plt.tight_layout
        plt.savefig('../imgs/high_of_day_results.png')
        plt.show();

    def last_30_days_plot_high(self):
        # Plot The Train Results
        plt.figure(figsize=(12, 6))
        plt.plot(self.result_df[-30:].index, self.result_df['y_hat_high'][-30:], c='r', marker='o', markersize=1, label='Prediction')
        plt.plot(self.result_df[-30:].index, self.result_df['tmr_high'][-30:], 'k--', label='True Value')
        plt.ylabel('Price ($)', fontsize=16, c='k')
        plt.xlabel('Daily Samples, Last 30 Market Days', fontsize=16, c='k')
        plt.xticks([])
        plt.title('Predictions and True High of Day Price', fontsize=19)
        plt.legend()
        plt.tight_layout
        plt.savefig('../imgs/high_of_day_results_30d.png')
        plt.show();

    def get_plots_low(self):
        # Plot The Train Results
        plt.figure(figsize=(12, 6))
        plt.plot(self.result_df.index, self.result_df['y_hat_low'], c='r', marker='o', markersize=1, label='Prediction')
        plt.plot(self.result_df.index, self.result_df['tmr_low'], 'k--', label ='True Value')
        plt.ylabel('Price ($)', fontsize=16, c='k')
        plt.xlabel('Daily Samples Back to January 2000', fontsize=16, c='k')
        plt.xticks([])
        plt.title('Predictions and True Low of Day Price', fontsize=19)
        plt.legend()
        plt.tight_layout
        plt.savefig('../imgs/low_of_day_results.png')
        plt.show();

    def last_30_days_plot_low(self):
        # Plot The Train Results
        plt.figure(figsize=(12, 6))
        plt.plot(self.result_df[-30:].index, self.result_df['y_hat_low'][-30:], c='r', marker='o', markersize=1, label='Prediction')
        plt.plot(self.result_df[-30:].index, self.result_df['tmr_low'][-30:], 'k--', label='True Value')
        plt.ylabel('Price ($)', fontsize=16, c='k')
        plt.xlabel('Daily Samples, Last 30 Market Days', fontsize=16, c='k')
        plt.xticks([])
        plt.title('Predictions and True Low of Day Price', fontsize=19)
        plt.legend()
        plt.tight_layout
        plt.savefig('../imgs/low_of_day_results_30d.png')
        plt.show();
    
    def create_save_15_day_dataframe(self):
        recent_resutls = self.result_df[-15:, :].copy()
        recent_resutls.to_csv('../data/15_day_results.csv')

    def get_predictions(self):
        self.column_manipulation()
        self.create_windows()
        self.train_test_holdout()
        self.parameter_selection_high()
        self.run_model_high()
        self.parameter_selection_low()
        self.run_model_low()
        self.create_result_dataframe()
        self.todays_predictions()
        self.get_plots_high()
        self.get_plots_low()
        self.last_30_days_plot_high()
        self.last_30_days_plot_low()

if __name__ == '__main__':
    # Create the date range for data
    start_date = '2000-01-01'
    end_date = date.today()
    # Ticker symbol to investigate
    ticker = 'SPY'
    # Pull data for both low and high of day
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    window = 10
    RandomForestPredict(data, window).get_predictions()