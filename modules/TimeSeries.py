import pandas as pd # 2.2.2
import numpy as np #1.26.4
from datetime import datetime

# statsmodels version 0.14.4
from statsmodels.tsa.arima.model import ARIMA


# Tensorflow 2.18.0
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, BatchNormalization, Activation

# sklearn 1.6.1
from sklearn.preprocessing import MinMaxScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Arch packages version 7.2.0
from arch import arch_model

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class timeseries():
    
    def __init__(self, maturity_time,
                 data    
                 ):
        self.T = maturity_time
        self.stock_data = data
        
    
    def arima(self):
        T = self.T
        
        train = self.stock_data.iloc[:-T]
        model = ARIMA(train.log_close.dropna().astype(float), order = (22,2,2) ) #order = (ar, diff, ma)
        model_fit = model.fit()
        
        return model_fit
    
    def garch(self):
        
        T = self.T
        split_date = datetime.strptime(self.stock_data[['log_close']].iloc[-T:-T+1].index[0], "%Y-%m-%d").date
        model_garch = arch_model(self.stock_data.log_close.diff().diff().dropna().astype(float), mean = 'Zero', vol='garch', p=1, o=0, q=1)
        model_garch_fit = model_garch.fit(last_obs=split_date)
       
        return model_garch_fit 
    
    def forecast(self):
        
        T = self.T
        model_arima = self.arima()
        model_vol   = self.garch()
        trend_pred = model_arima.forecast(steps = T)
        vol_pred = model_vol.forecast(start = len(self.stock_data.iloc[:-T-2]), horizon = T).variance
        stock_price_vol = pd.DataFrame({'trend':trend_pred.values, 'vol': vol_pred.iloc[:,0].T })
        
        stock_price_vol['high'] = stock_price_vol.apply(lambda x: x[0] * np.exp(-0.5 * x[1]**2 + x[1]) , axis = 1)
        stock_price_vol['low']  = stock_price_vol.apply(lambda x: x[0] * np.exp(-0.5 * x[1]**2 - x[1]) , axis = 1)
        
        return stock_price_vol


class Custom_RNN():
    def __init__(self,rnn_type, 
                 num_rnn_layers, 
                 seq_len, 
                 num_features, 
                 hidden_layer_shapes,
                 ls_activations = [None],
                initializer = 'glorot_normal',
                 **kwargs):
        self.rnn_type = rnn_type
        self.n_layers = num_rnn_layers
        self.seq_len  = seq_len
        self.num_feat = num_features
        self.acts     = ls_activations
        self.inits    = initializer
        self.nodes    = hidden_layer_shapes

    
    def Model(self):
        if self.rnn_type == 'LSTM':
            model = Sequential([
                LSTM(units = self.nodes[0], 
                     input_shape = (self.seq_len, self.num_feat),
                     kernel_initializer = self.inits,
                     return_sequences = True,
                    unroll=True),
                BatchNormalization(trainable=True, scale=True, center=True),
                Activation(self.acts[0])
            ])
            for i in range(1, self.n_layers):
                model.add(LSTM(units = self.nodes[i], kernel_initializer = self.inits,return_sequences = True))
                model.add(BatchNormalization(trainable=True, scale=True, center=True))
                model.add(Activation(self.acts[i]))
            model.add(Dense(units = 1))
        
        elif self.rnn_type == 'GRU':
            model = Sequential([
                GRU(units = self.nodes[0], 
                     input_shape = (self.seq_len, self.num_feat),
                     kernel_initializer = self.inits,
                     return_sequences = True,
                   unroll=True),
                BatchNormalization(trainable=True, scale=True, center=True),
                Activation(self.acts[0])
            ])
            for i in range(1, self.n_layers):
                model.add(GRU(units = self.nodes[i], kernel_initializer = self.inits,return_sequences = True))
                model.add(BatchNormalization(trainable=True, scale=True, center=True))
                model.add(Activation(self.acts[i]))
            model.add(Dense(units = 1))
        
        return model
    
    def train(self, data, T, epochs=200, batch_size = 30, lr = 2e-2):
        
        stock_data = data
        features = ['Close']
        X0 = stock_data[features] # Stock time series
        X0['log_price'] = X0['Close'].apply(lambda x: np.log(x))
        num_scaler = ('sc', MinMaxScaler())
        pipeline_num = Pipeline([num_scaler])
        transformer_num = [('num',pipeline_num, features )]
        ct = ColumnTransformer(transformers =  transformer_num)
        X_prep = ct.fit_transform(X0)
        
        num_steps  = self.seq_len + 1
        model_RNN  = self.Model()
        optimizer = tf.optimizers.Adam(learning_rate= lr, weight_decay=5e-4)
        model_RNN.compile(optimizer=optimizer, loss='Huber', metrics=['mse'])
        
        X = []; y = []

        for i in range(len(stock_data) - num_steps):
            X.append(X_prep[i:i+num_steps-1])
            y.append(X_prep[i+num_steps-1,0:])
        X = np.array(X).reshape(len(stock_data) - num_steps, -1, len(features) )
        y = np.array(y)

        #train data
        X_train, y_train = X[:-T], y[:-T]

        model_RNN.fit(X_train, y_train, 
                    validation_split = 0.1,
                    shuffle = False,
                    epochs=epochs, batch_size= batch_size,
                   verbose = 0)
        y_pred_train = model_RNN.predict(X_train)
        
        # forecasting

        y_RNN_forecast = [y_pred_train[-1:,:,0:1]]
        for i in range(T):
            pred = model_RNN(y_RNN_forecast[-1]).numpy().reshape(1,num_steps-1,1)
            y_RNN_forecast.append(pred)
        y_RNN_forecast = np.array(y_RNN_forecast)[1:,0,-1,0:1]
        y_RNN_forecast = MinMaxScaler().fit(X0['log_price'].values.reshape(-1,1)).inverse_transform(y_RNN_forecast)

        return y_RNN_forecast

            
       


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    