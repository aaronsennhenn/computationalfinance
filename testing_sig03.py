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



donchian_window_length = 20
adx_window_length = 18
prices = df_prices['AMD']
    
signals = pd.DataFrame(index=prices.index)
signals['signal'] = 0.0

adx = compute_adx(prices, adx_window_length)
donchian_sig, _ = donchian_signals(prices, donchian_window_length)
donchian_sig = np.asarray(donchian_sig)

#Custom tradig logic since adx only detects trends but not in which direction -> combine_two_subsignals() function doesn't work
position = np.zeros(len(prices), dtype=float)
holding = 0
for i in range(len(prices)):
   if np.isnan(adx[i]):
       continue
   if holding == 0 and (donchian_sig[i] == 1 or adx[i] > 25):
       holding = 1
   elif holding == 1 and donchian_sig[i] == 0 and adx[i] > 25:
       holding = 0
   position[i] = holding

signals['signal'] = position
signals['position_change'] = signals['signal'].diff().fillna(0)
signals.iloc[0, signals.columns.get_loc('position_change')] = 0



high, low = donchian_channel(prices, window_length=20)

aaa = pd.DataFrame({
    'adx': adx,
    'high': high,
    'low': low,
    'combined': position,
    'donchian': donchian_sig,
    'price': prices,
})