import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model

def create_dataset(data, look_back):
        dataX, datay = [], []
        for i in range(len(data)-look_back-1):
            a  = data[i:(i+look_back), 0:]
            dataX.append(a)
            datay.append(data[i + look_back, 0])
        return np.array(dataX), np.array(datay)

class PredictPriceHigh(object):
    def __init__(self, dataset):
        self.look_back = 2
        self.dataset = dataset.values
        self.dataset = self.dataset.astype('float32').reshape(-1, 1)
        self.train_size = int(len(self.dataset) * .67)
        self.train_x = dataset.index[0:self.train_size]
        self.test_x = dataset.index[self.train_size:len(self.dataset)-self.look_back]

    def normalize_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.scaler.fit_transform(self.dataset)

    def train_test_split(self):
        self.train = self.dataset[0:self.train_size, :]
        self.test = self.dataset[self.train_size:len(self.dataset)-self.look_back, :]
        self.holdout = self.dataset[len(self.dataset)-self.look_back:, :]

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
        model.fit(self.X_train, self.y_train, epochs=300, batch_size=1024, verbose=0)
        # Save Model 
        model.save('../models/high_model.h5')
        # make predictions
        trainPredict = model.predict(self.X_train)
        testPredict = model.predict(self.X_test)
        # Invert predictions
        self.trainPredict = self.scaler.inverse_transform(trainPredict)
        self.y_train = self.scaler.inverse_transform([self.y_train])
        self.testPredict = self.scaler.inverse_transform(testPredict)
        self.y_test = self.scaler.inverse_transform([self.y_test])
        # Calculate RMSE
        trainScore = math.sqrt(mean_squared_error(self.y_train[0], self.trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(self.y_test[0], self.testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

    def get_plots(self):
        # Plot The Train Results
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_x[self.look_back+1:], self.trainPredict, 'r--', label='Predicted')
        plt.plot(self.train_x[self.look_back+1:], self.y_train.reshape(-1, 1), 'k--', label='True Value')
        plt.ylabel('Price ($)', fontsize=16)
        plt.xlabel('Historical Dates', fontsize=16)
        plt.xticks([])
        plt.title('Train Data Predicted and True Tomorrow High Price', fontsize=19)
        plt.legend()
        plt.tight_layout
        #plt.savefig('../imgs/high_train_lstm_tuned.png')
        plt.show();

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(self.test_x[self.look_back+1:], self.testPredict, 'r--', label='Predicted')
        plt.plot(self.test_x[self.look_back+1:], self.y_test.reshape(-1, 1), 'k--', label='True Value')
        plt.ylabel('Price ($)', fontsize=16)
        plt.xlabel('Historical Dates', fontsize=16)
        plt.xticks([])
        plt.title('Test Data Predicted and True Tomorrow High Price', fontsize=19)
        plt.legend()
        plt.tight_layout
        #plt.savefig('../imgs/high_test_lstm_tuned.png')
        plt.show();
    
    def return_todays_pred(self):
        finalX = np.array([[self.holdout[0], self.holdout[1]]])
        finalX = np.reshape(finalX, (finalX.shape[0], 1, finalX.shape[1]))
        v = load_model('../models/high_model.h5')
        finalPredict = v.predict(finalX)
        finalPredict = self.scaler.inverse_transform(finalPredict)
        print(f"I am tomorrow's high: {finalPredict[0][0]:.2f}")
        
    def get_predictions(self):
        self.normalize_data()
        self.train_test_split()
        self.reshape_tr_test()
        self.make_predictions()
        self.return_todays_pred()
        

if __name__ == '__main__':
    pass