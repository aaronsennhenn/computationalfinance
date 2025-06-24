import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

def download_stock_price_data(tickers, start_date, end_date):    
    # Download adjusted close prices
    df_prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

    # Convert to NumPy
    prices_array = df_prices.to_numpy()
    
    # Compute daily returns manually: (P_t / P_{t-1}) - 1
    prev = (prices_array[1:] / prices_array[:-1]) - 1

    # Create a DataFrame with the correct index and column names
    df_price_changes = pd.DataFrame(prev, index=df_prices.index[1:], columns=df_prices.columns)

    # Add first row with 0s (no return on first day)
    df_price_changes = pd.concat([
        pd.DataFrame([[0]*df_price_changes.shape[1]], index=[df_prices.index[0]], columns=df_prices.columns),
        df_price_changes
    ])

    return df_prices, df_price_changes


def moving_average(prices, window_length):
        return np.convolve(prices, np.ones(window_length)/window_length, mode='same')

def ma_signal(series, short_window, long_window):
    signals = pd.DataFrame(index=series.index)
    signals['signal'] = 0.0

    prices_array = series.to_numpy()
    signals['short_ma'] = moving_average(prices_array, short_window)
    signals['long_ma'] = moving_average(prices_array, long_window)

    valid_range = max(short_window, long_window)
    signals.iloc[valid_range:, signals.columns.get_loc('signal')] = np.where(
        signals.iloc[valid_range:, signals.columns.get_loc('short_ma')] >
        signals.iloc[valid_range:, signals.columns.get_loc('long_ma')],
        1.0, 0.0
    )

    signals['position_change'] = signals['signal'].diff()
    signals.loc[series.index[0], 'position_change'] = 0
    return signals

# Metric functions
def returns(prices, positions):
    prices_array = prices.to_numpy()
    daily_returns = (prices_array[1:] / prices_array[:-1]) - 1
    strategy_returns = positions[:-1] * daily_returns
    return strategy_returns


def compute_total_trading_return(position, initial_cash=1.0):
    final_value = position.iloc[-1]['stock_value'] + position.iloc[-1]['cash'] #Trade all leftover cash
    return (final_value - initial_cash) / initial_cash

def mean_return(returns):
    return np.sum(returns)/len(returns)

def std_deviation(returns):
    mean = mean_return(returns)
    return np.sqrt(np.sum((returns-mean)**2)/len(returns))

def sharpe(returns, periods_per_year=252):
    mean = mean_return(returns)
    std  = std_deviation(returns) 
    sharpe_ratio = (mean * periods_per_year) / (std * np.sqrt(periods_per_year))
    return sharpe_ratio

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


# Grid search
tickers = ["AAPL",]
start = "2010-01-01"
end = "2021-12-31"

df_prices, df_changes = download_stock_price_data(tickers, start, end)
price = df_prices["AAPL"]
price_change = df_changes["AAPL"]



results = []
short_windows = range(20, 100)   
long_windows = range(100, 200)

for short_w in short_windows:
    for long_w in long_windows:
        if short_w >= long_w:
            continue
        cumret, sharpe_val, dd, vol = backtest_ma(price, short_w, long_w)
        results.append({
            "short": short_w,
            "long": long_w,
            "cumulative_return": cumret,
            "sharpe_ratio": sharpe_val,
            "max_drawdown": dd,
            "volatility"  : vol 
        })

