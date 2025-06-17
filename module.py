### IMPORTS

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data



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



def moving_average(prices, window_length):
    return np.convolve(prices, np.ones(window_length)/window_length, mode='same')



def ma_signal(series, short_window, long_window):
    # Init
    signals = pd.DataFrame(index = series.index)
    signals['signal'] = 0.0
    
    # Compute simple moving average
    signals['short_ma'] = moving_average(series, short_window)
    signals['long_ma'] = moving_average(series, long_window)

    #print(signals)
    
    # Compute signals 
    signals.iloc[short_window:, signals.columns.get_loc('signal')] = np.where(
        signals.iloc[short_window:, signals.columns.get_loc('short_ma')] > signals.iloc[short_window:, signals.columns.get_loc('long_ma')] , 1.0, 0.0)   
    signals['position_change'] = signals['signal'].diff()
    signals.loc[series.index[0], 'position_change'] = 0
    return signals


def simulate_single_stock_trading(df_position_changes, df_price_changes, df_prices, initial_cash=1.0, capital_fraction_per_trade=0.2):

    def open_trade(position, signal):
        stock_value, cash = position
        if signal <= 0:
            return np.array([stock_value, cash])
        allocated = cash * (1 - (1 - capital_fraction_per_trade) ** signal)
        return np.array([stock_value + allocated, cash - allocated])

    def hold_trade(position, price_change):
        return np.array([position[0] * price_change, position[1]])

    def close_trade(position, signal):
        stock_value, cash = position
        if signal < 0:
            return np.array([0.0, cash + stock_value])
        return position

    positions = []
    is_first = True

    for idx in df_position_changes.index:
        signal = df_position_changes.loc[idx, 'position_change']
        price_change = df_price_changes.loc[idx]

        if is_first:
            current_pos = open_trade(np.array([0.0, initial_cash]), signal)
            is_first = False
        else:
            current_pos = hold_trade(positions[-1], price_change)
            current_pos = close_trade(current_pos, signal)
            current_pos = open_trade(current_pos, signal)

        positions.append(current_pos)

    df_position = pd.DataFrame(positions, index=df_prices.index, columns=['stock_value', 'cash'])
    
    return df_position



def gridsearch_best_ma_params(df_prices, df_price_changes, short_range, long_range):
    
    best_ret = -np.inf
    params = []
    returns = []
    
    for short_window in short_range:
        for long_window in long_range:
            if short_window >= long_window:
                continue
            
            #Get signals and position changes
            signals = ma_signal(df_prices, short_window, long_window)
            df_position_changes = signals['position_change'].to_frame()
            
            #Run Simulation
            df_position = simulate_single_stock_trading(df_position_changes, df_price_changes, df_prices, capital_fraction_per_trade=1)
                      
            #Compute return based on signals
            ret = compute_total_trading_return(df_position)
            returns.append(ret)
            params.append((short_window, long_window))
            
            if ret > best_ret:
                best_ret = ret
                best_params = (short_window, long_window)
                best_position = df_position
    
    combinations_df = pd.DataFrame({'params': params,'return': returns})
                
    return best_params, best_ret, best_position, combinations_df



def compute_total_trading_return(df_position, initial_cash=1.0):
    final_value = df_position.iloc[-1]['stock_value'] + df_position.iloc[-1]['cash'] #Trade all leftover cash
    return (final_value - initial_cash) / initial_cash


############ Benchmark functions ################################################

def buy_and_hold_return(df_prices):
    return (df_prices['AAPL'].iloc[-1] - df_prices['AAPL'].iloc[0]) / df_prices['AAPL'].iloc[0]


def random_trading_return(df_prices, df_price_changes):
    np.random.seed(42)
    random_signals = np.random.choice([-1, 0, 1], size=len(df_prices), p=[0.1, 0.8, 0.1])
    df_random_position_change = pd.DataFrame(data=random_signals, index=df_prices.index, columns=['position_change']) #Random position changes
    df_position = simulate_single_stock_trading(df_random_position_change, df_price_changes, df_prices, capital_fraction_per_trade=1)    
    
    return compute_total_trading_return(df_position, initial_cash=1.0)


###############################################################################

