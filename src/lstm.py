import math
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

df = pd.read_csv('../data/lstm_testdata.csv')
df.set_index('index', inplace=True)
df.drop('tomorrow_high', axis=1, inplace=True)
col_name = 'high'
first_col = df.pop(col_name)
df.insert(0, col_name, first_col)
training_set = df.values
# Perform Scaling
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
sc_predict = MinMaxScaler(feature_range=(0, 1))
sc_predict.fit_transform(training_set[:, 0:1])
# Create a data structure with 30 timestamps and 1 output
X_train = []
y_train = []

n_future = 1 #number of days that I want to predict into the future
n_past = 30 # number of days in the past that I want to use for y_hat

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past: i, 0:df.shape[1] - 1])
    y_train.append(training_set_scaled[i + n_future: i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print('X_train shape is: {}'.format(X_train.shape))
print('y_train shape is: {}'.format(y_train.shape))

# Initializing the Neural Network Based On LSTM
model = Sequential()
# Adding 1st LSTM Layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, df.shape[1]-1)))
# Adding 2nd LSTM Layer
model.add(LSTM(units=10, return_sequences=False))
# Adding Dropout
model.add(Dropout(0.25))
# Output Layer
model.add(Dense(units=1, activation='relu'))
# Compiling the Neural Network
model.compile(optimizer ='adam', loss='mean_squared_error')

# START TRAINING
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')

history = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)