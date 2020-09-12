import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_windows(data, window, col):
        for i in range(window+1):
            data[col+'_'+str(i)] = data[col].shift(i, axis=0)

class PredictPrice(object):
    def __init__(self, dataset, pred_col, focus_col):
        self.dataset = dataset
        self.pred_col = pred_col
        self.focus_col = focus_col
        self.look_back = 5
        self.num_cols = len(self.dataset.columns) - 1
        # Remove the column opposite of the column of interest (reduce dimensions)
        if self.focus_col == 'high':
            self.dataset.drop('low', axis=1, inplace=True)
        elif self.focus_col == 'low':
            self.dataset.drop('high', axis=1, inplace=True)

    def train_test_split(self):
        self.y = self.dataset[self.pred_col]
        self.X = self.dataset.drop(columns=[self.pred_col], axis=1)
        # Create the look back window for the price associated with the target
        create_windows(self.X, self.look_back, self.focus_col)
        self.X.dropna(axis='rows', how='any', inplace=True)
        # Make sure that y shape matches X shape after removing NaN's
        idx = self.X.index
        self.y = self.y[idx]
        # Train/Test split for X and y
        train_size = int(len(self.dataset) * .67)
        self.X_train = np.array(self.X.iloc[0:train_size, :].values)
        self.X_test = np.array(self.X.iloc[train_size:len(self.dataset), :].values)
        self.y_train = np.array(self.y[0:train_size].values).reshape(-1, 1)
        self.y_test = np.array(self.y[train_size:].values).reshape(-1, 1)
    
    def scale_X_y(self):
        # Create the scaler objects for X and y
        self.y_scaler = MinMaxScaler()
        self.X_scaler = MinMaxScaler()
        # Train the scaler objects on only the train data
        self.y_scaler.fit(self.y_train)
        self.X_scaler.fit(self.X_train)
        # Transform
        self.y_train_sc = self.y_scaler.transform(self.y_train)
        self.X_train = self.X_scaler.transform(self.X_train)
        self.X_test = self.X_scaler.transform(self.X_test)
        
    def reshape_X(self):
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
    
    def make_predictions(self):
        # Initializing the Neural Network Based On LSTM
        model = Sequential()
        # Adding 1st LSTM Layer
        model.add(LSTM(units=640, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        
        # Adding 2nd LSTM Layer
        model.add(LSTM(units=320, return_sequences=True))
        model.add(Dropout(0.25))
        # # Adding 3rd LSTM Layer
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.25))
        # # Adding 4th LSTM Layer
        model.add(LSTM(units=10, return_sequences=False))
        model.add(Dropout(0.25))
        # Output Layer
        model.add(Dense(units=1, activation='relu'))
        # Compiling the Neural Network
        model.compile(loss='mean_squared_error', optimizer ='adam')
        # Fit on training data
        model.fit(self.X_train, self.y_train_sc, epochs=300, batch_size=1024, verbose=2)
        # make predictions
        trainPredict = model.predict(self.X_train)
        testPredict = model.predict(self.X_test)
        trainPredict.reshape(-1, 1)
        testPredict.reshape(-1, 1)
        # Invert predictions
        trainPredict = self.y_scaler.inverse_transform(trainPredict)
        testPredict = self.y_scaler.inverse_transform(testPredict)
        # Calculate RMSE
        trainScore = math.sqrt(mean_squared_error(self.y_train, trainPredict))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(self.y_test, testPredict))
        print('Test Score: %.2f RMSE' % (testScore))

    def get_predictions(self):
        self.train_test_split()
        self.scale_X_y()
        self.reshape_X()
        self.make_predictions()
        
if __name__ == '__main__':
    # Testing my model with a saved DataFrame
    np.random.seed(42)
    df = pd.read_csv('../data/lstm_testdata.csv')
    df.set_index('index', inplace=True)
    # Testing prediction class
    PredictPrice(df, 'tomorrow_high', 'high').get_predictions()


