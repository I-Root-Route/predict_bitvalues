import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf

import data_processing

dataset = data_processing.data_i

X = dataset.iloc[:,[4,5,6,8,9]].values #入力層
y = dataset.iloc[:,7].values #出力層
 
#テストデータと評価用データに分ける
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

#ANN構築
ann = tf.keras.models.Sequential()

#input layer and the hidden layers 2階層
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#output layer
ann.add(tf.keras.layers.Dense(units=1))

#Compiling the ANN
ann.compile(optimizer='adam', loss='mean_squared_error')

#Training the ANN model
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#Prediction
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))



