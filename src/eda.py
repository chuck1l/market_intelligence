# Standard Imports
import pandas as pd 
import numpy as np 
from datetime import date 
from datetime import timedelta 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
# Model Imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Source Data From Yahoo
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

class StockEda(object):
    def __init__(self, data, g_label):
        self.g_label = g_label
        self.df = data
        if self.g_label == 'tmr_high':
            self.plot_label = 'Tomorrow High'
        elif self.g_label == 'tmr_low':
            self.plot_label = 'Tomorrow Low'
        else:
            self.plot_label = ''

    def column_manipulation(self):
        self.df.columns = self.df.columns.str.lower()
        self.df['tmr_high'] = self.df['high'].shift(periods=-1)
        self.df['tmr_low'] = self.df['low'].shift(periods=-1)
        self.df.drop(self.df.tail(1).index, inplace=True)

    def ensure_float(self):
        cols = self.df.columns
        self.df[cols] = self.df[cols].astype(float).round(2)

    def create_X_y_train(self):
        self.y = self.df[self.g_label]
        self.X = self.df.copy()
        self.X.drop(columns=['tmr_high', 'tmr_low'], axis=1, inplace=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, shuffle=True, random_state=42)
        
    def feature_importance(self):
        labels = pd.Series(self.X_train.columns, name='features')
        # Using Random Forest Model with default inputs
        randomforest_model = RandomForestRegressor()
        randomforest_model.fit(self.X_train, self.y_train)
        # Predict for graph
        y_hat = randomforest_model.predict(self.X_test)
        # Get feature importances
        importances = pd.Series(randomforest_model.feature_importances_, name='Feature Importance')
        features_df = pd.concat([labels, importances], axis=1)
        features_df = features_df.sort_values(by='Feature Importance', ascending=False)
        graph_labels = list(features_df.features)
        x = np.arange(features_df.shape[0]) # label locations
        width = 0.35 # setting the width of each bar for the labels
        # Plotting the Random Forest Model Feature importance
        save_location = '../imgs/' + self.g_label + '_feature_importance.png'
        save_location2 = '../imgs/' + self.g_label + '_initial_prediction_vs_true.png'
        result_df = pd.DataFrame(data=self.y_test)
        result_df['y_hat'] = y_hat
        result_df.sort_index(inplace=True)
        toggle = True
        if toggle:
            fig, ax = plt.subplots(figsize=(18, 12))
            ax.bar(x, features_df['Feature Importance'], width, label=self.plot_label)
            ax.set_ylabel('Importance Score', fontsize=16, c='k')
            ax.set_title('{} Feature Importance - Random Forest'.format(self.plot_label))
            ax.set_xticks(x)
            ax.set_xticklabels(graph_labels, rotation=45, fontsize=16, c='k')
            ax.legend()
            fig.tight_layout()
            plt.savefig(save_location);
            # Plot The True Test vs Predicted Test Results
            plt.figure(figsize=(12, 6))
            plt.plot(result_df.index, result_df['y_hat'], c='r', marker='o', markersize=1, label='Prediction')
            plt.plot(result_df.index, result_df[self.g_label], 'k--', label='True Value')
            plt.ylabel('Price ($)', fontsize=16, c='k')
            plt.xlabel('Historical Dates', fontsize=16, c='k')
            plt.xticks([])
            plt.title('Pre-Tuning Predicted vs. True {} Price'.format(self.plot_label), fontsize=19)
            plt.legend()
            plt.tight_layout
            plt.savefig(save_location2);
            plt.show()       
        
    def get_principal_comp(self):
        self.column_manipulation()
        self.ensure_float()
        self.create_X_y_train()
        self.feature_importance()
       
if __name__ == '__main__':
    # Create the date range for data
    start_date = '2000-01-01'
    end_date = date.today()
    # Ticker symbol to investigate
    ticker = 'SPY'
    # Pull data for both low and high of day
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    StockEda(data, 'tmr_high').get_principal_comp()
    StockEda(data, 'tmr_low').get_principal_comp()