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
   if holding == 0 and (donchian_sig[i] == 1 and adx[i] > 25):
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


signals_df = signal01(prices, 50, 200, 12, 26, 9)


plot_buy_and_sell_signals(signals_df['macd_position_change'], test_df_prices, 'AMD', 'MACD Signal')
plot_buy_and_sell_signals_np(signals_df['macd_position_change'], test_df_prices, 'AMD', 'MACD Signal')


strat_return = strategy_returns(prices, signals_df['signal'])
cumulative_return(strat_return)

df_pos, returns = new_strategy_returns(prices, signals_df[['position_change']])

sma = moving_average(prices, window_length)

 stds = np.empty_like(prices)
    half_w = window_length // 2
    
    for i in range(len(prices)):
        start = max(0, i - half_w)
        end = min(len(prices), i + half_w + 1)
        stds[i] = np.std(prices[start:end])
    
    upper_band = sma + num_std * stds
    lower_band = sma - num_std * stds
    
    
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

y = moving_average(x, 2)
y, x
len(y)
len(x)


def moving_average(prices, window_length):
    cumsum = np.cumsum(np.insert(prices, 0, 0)) 
    result = (cumsum[window_length:] - cumsum[:-window_length]) / window_length
    return np.concatenate((np.full(window_length - 1, np.nan), result))