import os
from datetime import datetime, date, timedelta
import json
from flask import Flask, jsonify, render_template
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import datetime
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler



def sequence_creator(input_data, window):
    """
    直近window日のデータを元に(window_fr)、次の日(label)のデータを予測するため
    return -> (直近データ, 正解ラベル) (Tuple List)
    """
    dataset = []
    data_len = len(input_data)
    for i in range(data_len - window):
        window_fr = input_data[i:i+window]
        label = input_data[i+window:i+window+1]
        dataset.append((window_fr,label))
    return dataset

class LSTM_BITCOIN(nn.Module):
    def __init__(self, in_size=1, h_size=30, out_size=1):
        """
        in_size -> size of input layer
        h_size -> size of hidden layer
        out_size -> size of output layer
        """
        super().__init__()

        self.h_size = h_size
        self.lstm = nn.LSTM(in_size, h_size) #モデル形成
        self.fc = nn.Linear(h_size, out_size)

        self.hidden = (torch.zeros(1,1,self.h_size),torch.zeros(1,1,self.h_size))

    def forward(self, sequence_data):
        """
        予測をする時に呼び出す関数
        """
        lstm_out, self.hidden = self.lstm(sequence_data.view(len(sequence_data),1,-1), self.hidden)
        pred = self.fc(lstm_out.view(len(sequence_data),-1))
        return pred[-1] #最後の要素が予測値



if __name__ == "__main__":

    df = pd.read_csv('crypto_data2.csv',index_col=0)    #日付をインデックス化
    df = df["price_close"]                              #price_closeのみの予測
    #df.plot(figsize=(15,6),color="red")
    #plt.show()
    df.index = pd.to_datetime(df.index)
    y = df.values.astype(float)

    torch.manual_seed(3)

    model = LSTM_BITCOIN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #lr -> 学習率

    epochs = 100
    window_size = 7
    loss_list = []
    loss_temp = 0 #誤差の初期化


    """
    入力データセットの正規化
    """
    scaler = MinMaxScaler(feature_range=(-1,1)) #入力データ値が-1から1におさまるようにする
    y_normalized = scaler.fit_transform(y.reshape(-1,1))
    y_normalized = torch.FloatTensor(y_normalized).view(-1)
    full_data = sequence_creator(y_normalized,window_size) #return -> [(直近のデータ,正解ラベル)...]

    if not os.path.isfile("lstm_model.pth"):
        for epoch in range(epochs):
            for sequence_in, y_train in full_data:
                
                y_pred = model(sequence_in)
                
                loss = criterion(y_pred, y_train)
                loss_temp += loss
                
                optimizer.zero_grad()
                model.hidden = (torch.zeros(1,1,model.h_size),torch.zeros(1,1,model.h_size))
                
                loss.backward()
                optimizer.step()
                
            if((epoch+1) % 10 ==0):
                loss_list.append(loss_temp.item()/(10*len(full_data)))
                print(f'Epoch {epoch+1} Loss {loss_temp.item()/(10*len(full_data))}')
                loss_temp = 0
                
        torch.save(model.state_dict(), "lstm_model.pth")

    else:
        model.load_state_dict(torch.load("lstm_model.pth"))
        upcoming_future = 7
        predictions = y_normalized[-window_size:].tolist()

        today = datetime.datetime.today()
        string_today = datetime.datetime.strftime(today, '%Y-%m-%d')
        a_week_later = today + timedelta(days=upcoming_future)
        string_a_week_later = datetime.datetime.strftime(a_week_later, '%Y-%m-%d')

        model.eval()

        for i in range(upcoming_future):
            sequence = torch.FloatTensor(predictions[-window_size:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1,1,model.h_size),torch.zeros(1,1,model.h_size))
                predictions.append(model(sequence).item())
                    
        predictions_y = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        x = np.arange(string_today, string_a_week_later, dtype='datetime64[D]').astype('datetime64[D]')
        """
        sns.set()
        plt.figure(figsize = (12,4))
        plt.title("BTC/JPY")
        plt.grid(True)
        plt.ylabel("BTC/JPY")
        plt.plot(df)
        plt.plot(x,predictions_y[-window_size:])
        plt.show()
        """
        print(predictions_y[-window_size:])

        app = Flask(__name__)

        response = {}
        datelist = []
        for i in range(upcoming_future):
            datelist.append(datetime.datetime.strftime(today + timedelta(days=i), '%Y-%m-%d'))

        @app.route('/', methods=['GET'])
        def predicted_data():
            for i, date in enumerate(datelist):
                response[date] = predictions_y[-window_size:][i][0]
            
            return jsonify(response), 200
        app.run(debug=True)

        
