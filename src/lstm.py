import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class PredictPrice(object):
    def __init__(self, dataset, pred_col):
        self.dataset = dataset
        self.pred_col = pred_col
        self.look_back = 5
        self.num_cols = len(self.dataset.columns) - 1

    def train_test_split(self):
        self.y = self.dataset[self.pred_col]
        self.X = self.dataset.drop(columns=[self.pred_col], axis=1)
        # Train/Test split for X and y
        train_size = int(len(self.dataset) * .67)
        print(train_size)
        self.X_train = np.array(self.X.iloc[0:train_size, :].values)
        self.X_test = np.array(self.X.iloc[train_size:len(self.dataset), :].values)
        self.y_train = np.array(self.y[0:train_size].values)
        self.y_test = np.array(self.y[train_size:].values)
        
    def create_windows(self):
        pass

    def make_predictions(self):
        # Initializing the Neural Network Based On LSTM
        model = Sequential()
        # Adding 1st LSTM Layer
        model.add(LSTM(units=640, return_sequences=True, input_shape=(self.num_cols, self.look_back)))
        # Adding 2nd LSTM Layer
        model.add(LSTM(units=64))
        # Adding Dropout
        model.add(Dropout(0.25))
        # Output Layer
        model.add(Dense(units=1, activation='relu'))
        # Compiling the Neural Network
        model.compile(loss='mean_squared_error', optimizer ='adam')
        # Fit on training data
        model.fit(self.X_train_w, self.y_train, epochs=10, batch_size=1024, verbose=2)
        # make predictions
        trainPredict = model.predict(self.X_train_w)
        testPredict = model.predict(self.X_test_w)
        # Invert predictions
        # trainPredict = self.scaler.inverse_transform(trainPredict)
        # self.y_train = self.scaler.inverse_transform([self.y_train])
        # testPredict = self.scaler.inverse_transform(testPredict)
        # self.y_test = self.scaler.inverse_transform([self.y_test])
        # Calculate RMSE
        trainScore = math.sqrt(mean_squared_error(self.y_train[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(self.y_test[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

    def get_predictions(self):
        self.train_test_split()
        self.reshape_tr_test()
        #self.make_predictions()
        

if __name__ == '__main__':
    np.random.seed(42)
    df = pd.read_csv('../data/lstm_testdata.csv')
    df.set_index('index', inplace=True)
    # Testing prediction class
    PredictPrice(df, 'tomorrow_high').get_predictions()


'''
def normalize_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.scaler.fit_transform(self.dataset)

'''