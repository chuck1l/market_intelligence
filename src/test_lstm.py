import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
plt.style.use('ggplot')
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# For LSTM Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

np.random.seed(42)
df = pd.read_csv('../data/lstm_testdata.csv')
df.set_index('index', inplace=True)
train_size = int(len(df) * .67)
df_train = df.iloc[0:train_size, :]
df_test = df.iloc[train_size:, :]
# Scale Features
s1 = MinMaxScaler(feature_range=(-1,1))
Xs = s1.fit_transform(df_train[['high']])
# Scale Predictions 
s2 = MinMaxScaler(feature_range=(-1,1))
ys = s2.fit_transform(df_train[['high']])
# Each time step uses last 'window' to predict the next change
window = 5
X_train = []
y_train = []
for i in range(window, len(Xs)):
    X_train.append(Xs[i-window:i, :])
    y_train.append(ys[i])
# Reshape to a format accepted by LSTM
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# Create the LSTM model
model = Sequential() # Initialize LSTM model
model.add(LSTM(units=640, return_sequences=True, \
                input_shape=(1, window)))
model.add(Dropout(0.25))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.25))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error', \
                metrics='accuracy')
# Allow for early exit
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
# Fit LSTM model
history = model.fit(X_train, y_train, epochs=300, batch_size=1024, 
                    callbacks=[es], verbose=1)
# Save the model
model.save('../models/model.h5')
# Verify the fit of the model
trainPredict = model.predict(X_train)

# Un-scale the outputs
yu = s2.inverse_transform(trainPredict)
ym = s2.inverse_transform(y_train)
# Calculate Train RMSE
trainScore = math.sqrt(mean_squared_error(ym, yu))
print('Train Score: %.2f RMSE' % (trainScore))
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df_train.index[window:], yu, 'r--', label='Predicted')
plt.plot(df_train.index[window:], ym, 'k--', label='True Value')
plt.ylabel('Price ($)', fontsize=16)
plt.xlabel('Historical Dates', fontsize=16)
plt.xticks([])
plt.title('Train Data Predicted and True Tomorrow High Price', fontsize=19)
plt.legend()
plt.tight_layout
plt.savefig('../imgs/train_lstm_tuned.png')
plt.show();

# Load model
v = load_model('../models/model.h5')
Xt = df_test[['high']].values
yt = df_test[['high']].values
# Transform based on previous scaler
Xts = s1.transform(Xt)
yts = s2.transform(yt)
# Create the windows
X_test = []
y_test = []
for i in range(window, len(Xts)):
    X_test.append(Xts[i-window: i, :])
    y_test.append(yts[i])
# Reshape data to format accepted by LSTM
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# Verify the fit of the model
testPredict = model.predict(X_test)
# Un-scale outputs
ytu = s2.inverse_transform(testPredict)
ytm = s2.inverse_transform(y_test)
# Calculate Test RMSE
testScore = math.sqrt(mean_squared_error(ytm, ytu))
print('Test Score: %.2f RMSE' % (testScore))
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df_test.index[window:], ytu, 'r--', label='Predicted')
plt.plot(df_test.index[window:], ytm, 'k--', label='True Value')
plt.ylabel('Price ($)', fontsize=16)
plt.xlabel('Historical Dates', fontsize=16)
plt.xticks([])
plt.title('Test Data Predicted and True Tomorrow High Price', fontsize=19)
plt.legend()
plt.tight_layout
plt.savefig('../imgs/test_lstm_tuned.png')
plt.show();





