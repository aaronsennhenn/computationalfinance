## HINTS
### WRITE REUSABLE CODE
### USE NAMINGS THAT ARE EASY TO UNDERSTAND
### WRITE COMMENTS TO PROVIDE EXTRA CONTEXT
### ONLY USE NUMPY FOR PERFORMING NUMERICAL COMPUTATIONS


### IMPORTS

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data


### DOWNLOAD DATA

def download_stock_price_data(tickers, start_date, end_date):    
    # download the time series of adj. close price 
    # for each of the tickers from Yahoo finance
    # and dataframe with price changes
    #
    # in case you receive an error update your yfinance to the current version
    # and/or perform bug fixing
    
    df_prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    prev = df_prices.to_numpy()
    prev = prev / np.insert(prev[:-1,:], 0, np.ones(prev.shape[1]), 0)
    prev[0] = np.ones(prev.shape[1])
    df_price_changes = df_prices.copy(deep=True)
    df_price_changes[:] = prev

    return df_prices, df_price_changes


### HELPER FUNCTIONS USED ACROSS SIGNALS (in case needed)

def moving_average(prices, window_length):
    return np.convolve(prices, np.ones(window_length)/window_length, mode='same')

### Metrics used to evaluate trading signals


def mean_return(returns):
    return np.sum(returns) / len(returns)

def std_deviation(returns):
    mean = mean_return(returns)
    std = std_deviation(returns)
    return np.sqrt(np.sum((returns-mean)**2) / len(returns))

def sharpe(returns, periods_per_year=252):
    mean = mean_return(returns)
    std = std_deviation(returns)
    return (mean * periods_per_year) / (std * np.sqrt(periods_per_year))

def cumulative_return(returns):
    return np.prod(1 + returns) - 1

def max_drawdown(returns):
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = 1 - cum_returns / peak
    return np.max(drawdown)

def volatility(returns):
    return np.sqrt(np.sum((returns-np.mean(returns))**2) / len(returns))


def backtest_ma(price, short_window, long_window):
    signals = ma_signal(price, short_window, long_window)
    position = signals['signal'].shift(1).fillna(0).to_numpy()
    
    strat_returns = returns(price, position)
    cumret = cumulative_return(strat_returns)
    sharpe_val = sharpe(strat_returns) 
    max_dd = max_drawdown(strat_returns)
    volatility = std_deviation(strat_returns)
    return cumret, sharpe_val, max_dd, volatility
    


### MOVING AVERAGE SIGNAL

# related literature / PLEASE REPLACE THIS BY THE LITERATURE THAT YOUR SIGNAL IS BASED ON:
## Marshall, Ben R.; Nguyen, Nhut H. & Visaltanachoti, Nuttawat.
## “Time series momentum and moving average trading rules”
## Published in Quantitative Finance, Vol. 17, Issue 3, pp. 405-521, (2017).

def ma_signal(series, short_window, long_window):
    # Init
    signals = pd.DataFrame(index = series.index)
    signals['signal'] = 0.0
    
    # Compute simple moving average
    short_window = 250
    long_window = 500
    signals['short_ma'] = moving_average(series, short_window)
    signals['long_ma'] = moving_average(series, long_window)

    print(signals)
    
    # Compute signals 
    signals.iloc[short_window:, signals.columns.get_loc('signal')] = np.where(
        signals.iloc[short_window:, signals.columns.get_loc('short_ma')] > signals.iloc[short_window:, signals.columns.get_loc('long_ma')] , 1.0, 0.0)   
    signals['position_change'] = signals['signal'].diff()
    signals.loc[series.index[0], 'position_change'] = 0
    return signals