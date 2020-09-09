import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(data, look_back):
        dataX, datay = [], []
        for i in range(len(data)-look_back-1):
            a  = data[i:(i+look_back), 0:]
            dataX.append(a)
            datay.append(data[i + look_back, 0])
        return np.array(dataX), np.array(datay)

class PredictPrice(object):
    def __init__(self, dataset):
        self.look_back = 5
        self.dataset = dataset.values
        self.dataset = self.dataset.astype('float32').reshape(-1, 1)

    def normalize_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.scaler.fit_transform(self.dataset)

    def train_test_split(self):
        train_size = int(len(self.dataset) * .67)
        #test_size = len(self.dataset) - train_size
        self.train = self.dataset[0:train_size, :]
        self.test = self.dataset[train_size:len(self.dataset), :]

    def reshape_tr_test(self):
        X_train, self.y_train = create_dataset(self.train, self.look_back)
        X_test, self.y_test = create_dataset(self.test, self.look_back)
        self.X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        self.X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        
    def make_predictions(self):
        # Initializing the Neural Network Based On LSTM
        model = Sequential()
        # Adding 1st LSTM Layer
        model.add(LSTM(units=640, return_sequences=True, input_shape=(1, self.look_back)))
        # Adding 2nd LSTM Layer
        model.add(LSTM(units=64))
        # Adding Dropout
        model.add(Dropout(0.25))
        # Output Layer
        model.add(Dense(units=1, activation='relu'))
        # Compiling the Neural Network
        model.compile(loss='mean_squared_error', optimizer ='adam')
        # Fit on training data
        model.fit(self.X_train, self.y_train, epochs=300, batch_size=1024, verbose=2)
        # make predictions
        trainPredict = model.predict(self.X_train)
        testPredict = model.predict(self.X_test)
        # Invert predictions
        #breakpoint()
        trainPredict = self.scaler.inverse_transform(trainPredict)
        self.y_train = self.scaler.inverse_transform([self.y_train])
        testPredict = self.scaler.inverse_transform(testPredict)
        self.y_test = self.scaler.inverse_transform([self.y_test])
        # Calculate RMSE
        trainScore = math.sqrt(mean_squared_error(self.y_train[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(self.y_test[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

    def get_predictions(self):
        self.normalize_data()
        self.train_test_split()
        self.reshape_tr_test()
        self.make_predictions()
        

if __name__ == '__main__':
    np.random.seed(42)
    df = pd.read_csv('../data/lstm_testdata.csv')
    df.set_index('index', inplace=True)
    df.drop('tomorrow_high', axis=1, inplace=True)
    col_name = 'high'
    first_col = df.pop(col_name)
    df.insert(0, col_name, first_col)
    # Testing prediction class
    PredictPrice(df['high']).get_predictions()

    