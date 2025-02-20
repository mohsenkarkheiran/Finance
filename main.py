import streamlit as st

# my custom modules

from modules.TimeSeries import timeseries
from modules.TimeSeries import Custom_RNN
from modules.FQI import FQI

# frequently used modules
import pandas as pd # 2.2.2
import numpy as np #1.26.4

import tensorflow as tf # '2.16.2'
import keras # 3.7.0
from keras.layers import Dense
#from datetime import datetime as dt
import matplotlib.pyplot as plt #3.10.0
import time

from scipy.stats import norm # '1.15.1'

# Ouput plots:
def GBM(trend_pred, vol_pred, label = " "):
    stock_price_GBM = pd.DataFrame({'trend':trend_pred, 'vol': vol_pred})
    stock_price_GBM['high'] = stock_price_GBM.apply(lambda x: x[0] * np.exp(-0.5 * x[1] + np.sqrt(x[1]) ) , axis = 1)
    stock_price_GBM['low']  = stock_price_GBM.apply(lambda x: x[0] * np.exp(-0.5 * x[1] - np.sqrt(x[1]) ) , axis = 1)


    ax.plot(stock_price_GBM.index, stock_price_GBM['trend'], '-', label = label)
    ax.fill_between(stock_price_GBM.index, stock_price_GBM['high'], stock_price_GBM['low'], alpha = 0.1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock_price(USD)')
    ax.set_title(f'Price forecast for the next {T} days')
   
    


# Download the AAPL data:
stock_data = pd.read_csv("AAPL_raw.csv")
stock_data['Close'] = stock_data['4. close']
stock_data.set_index('date',inplace = True)
stock_data.sort_index(ascending = True,inplace = True)
stock_data['log_close'] = np.log(stock_data.Close)

###############################################################################

st.header("Call Option Analnysis", divider = 'blue')

###############################################################################

st.subheader("Part 1: Forecast stock-prices:")
# Collect data from user 
T = st.number_input(
    label = "Maturity time",
    min_value = 4,
    max_value = 50,
    value = 20)
RNN_type = st.selectbox(
    "Please select the RNN type",
    ('GRU', 'LSTM')
    )
num_rnn_layers= st.number_input(
    label = "number of RNN layers (integer)",
    min_value = 1,
    max_value = 3,
    value = 1
    )
hidden_layer_shapes = []
for i in range(int(num_rnn_layers)):
    layer_size = st.number_input(
        label= f'Please give me the size of the {i+1}th RNN layer',
        min_value = 10,
        max_value = 50,
        value = 30
        )
    hidden_layer_shapes.append( int(layer_size) )

seq_len = st.number_input(
    label = "Length of the sequence the you want to be processed by RNN layer(s)",
    min_value=1,
    max_value=100,
    value = 6
    )
epochs = st.number_input(
    label = 'How many epochs for training?',
    min_value = 50,
    max_value = 500,
    value = 200
    )
batch = st.number_input(
                        label = 'How many sequences do you to be processed at the same time?',
                        min_value = 1,
                        max_value = 50,
                        value = 30
                        )
lr = st.slider(label = "Determine the learning rate (in unite of 10^(-3))", 
               
               min_value=1,
               max_value=100,
               value = 20)

placeholder = st.empty()



if st.button("Train"):
    
    
    # CALL ARIMA
    placeholder.write(f"I'm using ARIMA and GARCH to forecast the stock price in the next {T} days. It may take a few minutes")
    T = int(T)
    time_series = timeseries(maturity_time = T, data = stock_data)
    forecasts = time_series.forecast()
    placeholder.empty()
    
    # CaLL RNN
    placeholder.write(f"I just configured the neural net, and I'm training it. I'll privde the next {T} days forecasts soon. Please be patient")
    model_RNN = Custom_RNN(rnn_type = RNN_type,
                       num_rnn_layers = int(num_rnn_layers), 
                       seq_len = int(seq_len ), 
                       num_features = 1, 
                       hidden_layer_shapes = hidden_layer_shapes )

    RNN_forecast = model_RNN.train(data = stock_data, T = T, 
                               epochs=int(epochs), batch_size = int(batch), lr = lr*0.001)

    placeholder.empty()
    forecasts['RNN_forecasts'] = RNN_forecast
    forecasts['Actual_data']=stock_data[['log_close']].iloc[-T:].copy()

   

    fig, ax = plt.subplots(figsize = (8,8))
    #GBM(forecasts.Actual_data, forecasts.vol, label = "Actual Prices")
    GBM(forecasts.trend, forecasts.vol, label='ARIMA Forecasted Prices')
    GBM(forecasts.RNN_forecasts, forecasts.vol , label='RNN Forecasted Prices')
    ax.plot(forecasts.index ,forecasts.Actual_data, label = "Actual Prices")
    ax.legend()
    indices = [dt[0] for dt in forecasts.index]
    ax.set_xticklabels(indices, rotation='vertical')
    
    
    st.pyplot(fig)

## Call PINN results
## I upload the weights of a trained NN onto the following neural net of the same configuration

#==============================================================================#
class PINN_Skeleton(keras.Model):

    def __init__(self,
                 layer_sizes=[1, 32, 32, 1],
                 activations=[None, "tanh", "tanh", "linear"],
                 kernel_initializer='glorot_normal',
                 parameters=[],
                 **kwargs):
       
        super().__init__(**kwargs)
        assert len(activations) == len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.input_dim   = layer_sizes[0]
        self.output_dim  = layer_sizes[-1]
        self.parameters  = parameters

        self.model_layers = [Dense(nu, activation=act, kernel_initializer=kernel_initializer)
                             for nu, act in zip(self.layer_sizes[1:], self.activations[1:])]
        self.build(input_shape=(None, self.input_dim))

    def call(self, X):
        
        if (func := self.activations[0]) is not None:  # apply input transformation
          X = keras.activations.get(func)(X)

        for layer in self.model_layers:
            X = layer(X)
        return X

vol = keras.Variable(.022, trainable = True, dtype = 'float64', name = 'volatility' ) # implied daily volatility, found by PINN
r = 4.19/252 # daily risk free rate, found from IRS
K =120. # the assumed strike price

def input_transform(X): # shape (N, 4) The custom activation function of the input layer. 

    X = tf.cast(X, dtype = 'float64')
    x = X[:,0:1]
    t = X[:,1:2]
    vol_node = X[:,2:3] * vol
    r_node  = X[:,3:] * r
    X_out = tf.concat( [x,t,vol_node,r_node], axis=-1)
    return X_out

BS_pinn =  PINN_Skeleton(layer_sizes=[4, 50, 50, 50, 1],
                    activations=[input_transform , 'tanh', 'tanh', 'tanh', 'relu'],
                    kernel_initializer='glorot_normal',
                     parameters = [vol])

# initialize the model
X = np.array([[ 1., 1.,  1.,  1. ]])
X_in = tf.cast(X,dtype = 'float64')
test = BS_pinn(X_in)
BS_pinn.load_weights("BS_pinn.weights.h5")


def Greeks(model, X):
    
    x, t, vol_node, r_node = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:,3:]  # shapes (N, 1)
    xtvr = [x, t, vol_node, r_node]
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xtvr)
        X = tf.concat(xtvr, axis=-1)  # shape (N, 4)
        
        V = model(X, training=True)  # shape (N, 1), training=True
        V_x, V_t, Vega, rho = tape.gradient(V, xtvr)  # shape (N, 1)
        V_xx= tape.gradient(V_x, x)
       

    V_S = (1/K)*tf.math.exp(-x) * V_x
    V_SS= ((1/K)**2) * tf.math.exp(-2*x) * (V_xx - V_x)
    greeks = {'Delta':V_S, 'Gamma': V_SS, 'theta': V_t, 'Vega': (1/vol) * Vega}

    return greeks

#==============================================================================#
st.subheader("PINN analysis")
if st.button("Show PINN results"):
    


    placeholder = st.empty()
    placeholder.write(f"PINN is trained for maturity time = 20, the daily implied volatility found by PINN is {vol}")
    
    time.sleep(5)
    
    placeholder = st.empty()
    placeholder.write("Once you give me the time and stock price, I'll compute the option price and Greeks...")
    
    

    S = st.number_input(
                    label = "Stock price (USD)",
                    min_value = 120.,
                    max_value = np.exp(1.) * 120,
                    value = 150.)

    t = st.number_input(
            label = "time (until maturity)",
            min_value = 0,
            max_value = 20,
            value = 1)

    x = np.log(S/120)

    X = np.array([[ x, t,  1.,  1. ]])
    X_in = tf.cast(X,dtype = 'float64')
    greeks = Greeks(BS_pinn, X_in)
    V = BS_pinn(X_in)
    
    st.subheader(f"Call optime price = {round(V, 2)}")
    st.subheader(f"Delta = {round(greeks['Delta'].numpy()[0][0], 2)}")
    st.subheader(f"Gamma = {round(greeks['Gamma'].numpy()[0][0], 2)}")
    st.subheader(f"theta = {round(greeks['theta'].numpy()[0][0], 2)}")
    st.subheader(f"Vega = {round(greeks['Vega'].numpy()[0][0], 2)}")

###############################################################################

st.subheader("FQI analysis")
if st.button('FQI results for option pricing'):
    
    placeholder = st.empty()
    placeholder.write("Monte-Carlo is used to simulate the future Stock market")
    
    time.sleep(5)
    
    placeholder = st.empty()
    placeholder.write("Give me the values of initial stock price, and Monte-Carlo parameters:")
    
    time.sleep(5)
    
    placeholder = st.empty()
    placeholder.write(f"Note: Smaller maturity times are preffered, strike price is assumed to be {120} for simplicity.")

    S0 = st.number_input(
        label = "Stock price now",
        min_value = 120.,
        max_value = np.exp(1.) * 120,
        value = 150.
        )

    M = st.number_input(
        label = "Maturity time",
        min_value = 1,
        max_value = 20,
        value = 2
        )

    steps = st.number_input(
        label = "Number of time steps",
        min_value = 5,
        max_value = 100,
        value = 7
        )

    N_MC = st.number_input(
        label = "Number of time steps",
        min_value = 1000,
        max_value = 50000,
        value = 10000
        )


    fqi = FQI( S0 = S0,
              sigma = 0.024, # Implied volatility found in PINN
              M = M,
              T = steps,
              r = 0.05,
              mu = 0.05,
              N_MC = N_MC,
              K = 120.,
              risk_lambda = 0.001)


    # The Black-Scholes prices
    def bs_call(t, S0=S0, K=K, r=r, sigma=0.024, T=M):
        d1 = (np.log(S0/K) + (r + 1/2 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
        d2 = (np.log(S0/K) + (r - 1/2 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
        price = S0 * norm.cdf(d1) - K * np.exp(-r * (T-t)) * norm.cdf(d2)
        return price


    # Monte-Carlo Stock prices
    S = fqi.Monte_Carlo()[0]

    # Optimal Hedges 
    a, Pi = fqi.optimal_hedge()

    # Optimal Q_DP
    Q = fqi.Optimal_Q_DP()

    # Prepare off-policy data for Q-learning
    fqi.off_policy_data()
    Q_RL, a_RL  = fqi.Optimal_Q_RL()





    ##################### PLOTS of FQI/DP ##########################
    step_size = N_MC // 10
    idx_plot = np.arange(step_size, N_MC, step_size)

    
    fig, ax = plt.subplots(3,2, figsize = (24,16))
    ax[0,0].plot(S.T.iloc[:, idx_plot])
    ax[0,0].set_xlabel('Time Steps')
    ax[0,0].set_title('Stock Price Sample Paths')
        
    ax[2,0].plot(a.T.iloc[:,idx_plot])
    ax[2,0].set_xlabel('Time Steps')
    ax[2,0].set_title('Optimal Hedge DP approach')

    ax[0,1].plot(Pi.T.iloc[:,idx_plot])
    ax[0,1].set_xlabel('Time Steps')
    ax[0,1].set_title('Portfolio Value')

    ax[1,1].plot(Q.T.iloc[:, idx_plot])
    ax[1,1].set_xlabel('Time Steps')
    ax[1,1].set_title('Optimal Q-Function ( = - C^(ask) with DP approach)')
    
    ax[1,0].plot(Q_RL.T.iloc[:, idx_plot])
    ax[1,0].set_xlabel('Time Steps')
    ax[1,0].set_title('Optimal Q-Function ( = - C^(ask) with FQI approach)')
    
    ax[2,1].plot(a_RL.T.iloc[:, idx_plot])
    ax[2,1].set_xlabel('Time Steps')
    ax[2,1].set_title('Optimal Hedging RL approach)')

    st.pyplot(fig)



    # QLBS option price
    C_QLBS = - Q_RL.copy() # Q_RL # 

    st.subheader('---------------------------------')
    st.subheader('       QLBS RL Option Pricing       ')
    st.subheader('---------------------------------\n')
    st.subheader('%-25s' % ('Initial Stock Price:'), S0)

    st.subheader('%-25s' % ('Strike:'), K)
    st.subheader('%-25s' % ('Maturity:'), M)
    st.subheader('%-26s %.4f' % ('\nThe QLBS Call Price 1 :', (np.mean(C_QLBS.iloc[:,0]))))
    st.subheader('%-26s %.4f' % ('\nBlack-Sholes Call Price:', bs_call(0)))
    st.subheader('\n')




















