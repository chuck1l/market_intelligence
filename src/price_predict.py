import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

class PredictPrice(object):
    def __init__(self, data):
        self.df = data
    def ensure_float(self):
        self.df.drop('date', axis=1, inplace=True)
        cols = self.df.columns
        self.df[cols] = self.df[cols].astype(float).round(2)
    def normalize_features(self):
        cols = self.df.columns
        scaler = StandardScaler()
        self.scaled = self.df.copy()
        self.scaled[cols] = scaler.fit_transform(self.scaled[cols])
    # Train on multiple Lag Timesteps
    def prediction(self):
        self.ensure_float()
        self.normalize_features()
        print(self.scaled)


if __name__ == '__main__':
    data = pd.read_csv('../data/lstm_test.csv')
    data.set_index('index', inplace=True)
    spy = PredictPrice(data)
    spy.prediction()

