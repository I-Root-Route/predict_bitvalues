import math
#import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import data_processing

df = data_processing.data_i

data = df.filter(['price_close'])

dataset = data.values

training_data_len = math.ceil(len(dataset)*0.8)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the training data set

train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train data sets
X_train = []
y_train = []

for i in range(10, len(train_data)):
    X_train.append(train_data[i-10:i,0])
    y_train.append(train_data[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Biuld LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

#Create the testing dataset
test_data = scaled_data[training_data_len - 10: , :]
X_test = []
y_test = dataset[training_data_len:, :]

for i in range(10, len(test_data)):
    X_test.append(test_data[i-10:i, 0])

#Convert the data to a numpy array
X_test = np.array(X_test)

#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#get the models predicted price values
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

#get the root mean squeared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#VIsualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price JPY', fontsize=18)
plt.plot(train['price_close'])
plt.plot(valid[['price_close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

