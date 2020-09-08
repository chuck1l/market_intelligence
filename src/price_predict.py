import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

class PredictPrice(object):
    def __init__(self, data, predict_col, lookback=5):
        self.df = data
        self.lookback = lookback
        self.predict_col = predict_col
    def ensure_float(self):
        cols = self.df.columns
        self.df[cols] = self.df[cols].astype(float).round(2)
    def normalize_features(self):
        cols = self.df.columns
        scaler = StandardScaler()
        self.scaled = self.df.copy()
        self.scaled[cols] = scaler.fit_transform(self.scaled[cols])
    # Create X, y with sliding windows of lookback period as specified above
    def X_y_windows(self):
        self.y = self.scaled[self.predict_col]
        self.X = self.scaled.drop(self.predict_col, axis=1)
        cols = self.X.columns
        for i in range(1, self.lookback + 1):
            for col in cols:
                self.X[col+'_'+str(i)] = self.X[col].shift(i)
        self.X.fillna(0, axis=0, inplace=True)
        ho_n = int(self.X.shape[0] * .1)
        tr_n = int(self.X.shape[0] * .8)
        self.X_train, self.X_test, self.X_holdout = self.X[:tr_n], self.X[tr_n:-ho_n], self.X[-ho_n:]
        self.y_train, self.y_test, self.y_holdout = self.y[:tr_n], self.y[tr_n:-ho_n], self.y[-ho_n:]
        #return self.X_train, self.y_train, self.X_test, self.y_test, self.X_holdout, self.y_holdout
    def run_lstm(self):
        
    def prediction(self):
        self.ensure_float()
        self.normalize_features()
        self.X_y_windows()

if __name__ == '__main__':
    pass
    # data = pd.read_csv('../data/lstm_test.csv')
    # data.set_index('index', inplace=True)
    # spy = PredictPrice(data)
    # spy.prediction()

