#Load packages
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas_datareader import data
import sys
sys.path.insert(0, r'C:\Users\aaron\OneDrive\Desktop\Data Science Studium\SS 25\Computational Python')
from module import *


#Load data in for in-sample
tickers = ["TSLA", "KHC", "AMD"]
start = "2020-01-01"
end = "2024-12-31"
df_prices, df_changes = download_stock_price_data(tickers, start, end)

#Load data in for out-of-sample
test_start = "2020-01-01"
test_end = "2024-12-31"
test_df_prices, test_df_changes = download_stock_price_data(tickers, test_start, test_end)




#params
rsi_window_length = 14
lower_rsi_bound = 30
upper_rsi_bound = 60
bollinger_window_length = 14
bollinger_n_stds = 1
prices = test_df_prices['KHC']


rsi = compute_rsi(prices, rsi_window_length)
sma, upper_band, lower_band = compute_bollinger_bands(prices, bollinger_window_length, bollinger_n_stds)

#RSI Signal
rsi_sig, _ = signal_rsi(prices, rsi_window_length, lower_rsi_bound, upper_rsi_bound)

#Bollinger Signal
bollinger_sig, _ = signal_bollinger(prices, bollinger_window_length, bollinger_n_stds)

#Combine Signals
combined = combine_two_subsignals(rsi_sig, bollinger_sig)

#Return combined signal dataframe
signals = pd.DataFrame(index=prices.index)
signals['signal'] = combined
signals['position_change'] = signals['signal'].diff().fillna(0)


aaa = pd.DataFrame({
    'RSI_Signal': rsi_sig,
    'Bollinger_Signal': bollinger_sig,
    'Combined_Signal': combined,
    'RSI': rsi,
    'upper': upper_band,
    'lower': lower_band,
    'price': prices,
})