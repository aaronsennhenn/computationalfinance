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

######## Signal Helper Function ############################

def combine_two_subsignals(signal1, signal2):
    
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)
    assert signal1.shape == signal2.shape

    combined = np.zeros_like(signal1)
    holding = 0

    for i in range(len(signal1)):
        if holding == 0 and (signal1[i] == 1 or signal2[i] == 1):
            holding = 1
        elif holding == 1 and signal1[i] == 0 and signal2[i] == 0:
            holding = 0
        combined[i] = holding

    return combined

###########################################################



######## Signal 01 ##################################################

def moving_average(prices, window_length):
    return np.convolve(prices, np.ones(window_length)/window_length, mode='same')

def ma_signal(series, short_window, long_window):
    signals = pd.DataFrame(index=series.index)
    signals['MA_signal'] = 0.0

    prices_array = series.to_numpy()
    signals['short_ma'] = moving_average(prices_array, short_window)
    signals['long_ma'] = moving_average(prices_array, long_window)

    valid_range = max(short_window, long_window)
    signals.iloc[valid_range:, signals.columns.get_loc('MA_signal')] = np.where(
        signals.iloc[valid_range:, signals.columns.get_loc('short_ma')] >
        signals.iloc[valid_range:, signals.columns.get_loc('long_ma')],
        1.0, 0.0)

    signals['position_change'] = signals['MA_signal'].diff()
    signals.loc[series.index[0], 'position_change'] = 0

    return signals['MA_signal'], signals


#Helper function exponentail moving avaerage
def exponential_moving_average(prices, MACD_window_length):

    ema = np.empty(len(prices))
    ema[:] = np.nan  # Fill with NaNs

    alpha = 2 / (MACD_window_length + 1)

    # Find first non-NaN index
    first_index = np.where(~np.isnan(prices))[0][0]

    start = first_index + MACD_window_length - 1

    if start < len(prices):
        # Compute initial average manually
        initial_sum = 0
        count = 0
        for i in range(start - MACD_window_length + 1, start + 1):
            if not np.isnan(prices[i]):
                initial_sum += prices[i]
                count += 1
        initial_avg = initial_sum / count
        ema[start] = initial_avg

        # Compute EMA recursively
        for i in range(start + 1, len(prices)):
            if not np.isnan(prices[i]):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def signal_macd(prices, short_window, long_window, signal_window):
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0

    price_array = prices.to_numpy()
    signals['short_ema'] = exponential_moving_average(price_array, short_window)
    signals['long_ema'] = exponential_moving_average(price_array, long_window)

    signals['MACD'] = signals['short_ema'] - signals['long_ema']
    signals['signal_line'] = exponential_moving_average(signals['MACD'].to_numpy(), signal_window)

    #Generate signal where MACD > signal line
    valid_range = max(short_window, long_window)
    signals.loc[signals.index[valid_range:], 'signal'] = np.where(
        signals['MACD'][valid_range:] > signals['signal_line'][valid_range:], 1.0, 0.0
    )

    signals['position_change'] = signals['signal'].diff()
    signals.loc[prices.index[0], 'position_change'] = 0

    return signals['signal'], signals

def signal01(prices, short_ma, long_ma, short_macd, long_macd, signal_window_macd):

    #MA Signal
    ma_sig, _ = ma_signal(prices, short_ma, long_ma)

    #MACD Signal
    macd_sig, _ = signal_macd(prices, short_macd, long_macd, signal_window_macd)

    #Combine Signals
    combined = combine_two_subsignals(ma_sig, macd_sig)

    #Return combined signal dataframe
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = combined
    signals['position_change'] = signals['signal'].diff().fillna(0)

    return signals

#######################################################################

############# Signal 02 ##############################################


def compute_rsi(prices, window_length):
    
    prices = np.asarray(prices).flatten()
    deltas = np.diff(prices)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    #Initilize and don't compute the first window_length days avg gain and loss
    avg_gain = np.empty_like(prices, dtype=float)
    avg_loss = np.empty_like(prices, dtype=float)    
    avg_gain[:window_length] = np.nan
    avg_loss[:window_length] = np.nan
    
    #First average gain and loss (simple mean)
    avg_gain[window_length] = gains[:window_length].mean()
    avg_loss[window_length] = losses[:window_length].mean()
    
    #Implement the Wilder smoothing after the first mean computation
    for i in range(window_length + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (window_length - 1) + gains[i - 1]) / window_length
        avg_loss[i] = (avg_loss[i - 1] * (window_length - 1) + losses[i - 1]) / window_length
    
    #Compute index
    rs =  avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    #Let the first window_length entries be nan, so it doens't generate any signal
    rsi[:window_length] = 50
    
    return rsi


def signal_rsi(prices, rsi_window, lower_rsi_bound, upper_rsi_bound):
    
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0

    pricnes_array = np.as_array(prices)
    rsi = compute_rsi(prices, rsi_window)
    
    buy_signal = price_array < lower_rsi_bound
    sell_signal = price_array > upper_rsi_bound

    #Create position signal with holding logic
    position = np.zeros(len(price_array), dtype=float)
    holding = 0
    for i in range(len(prices)):
        if holding == 0 and buy_signal[i]:
            holding = 1
        elif holding == 1 and sell_signal[i]:
            holding = 0
        position[i] = holding

    signals['signal'] = position
    signals['position_change'] = signals['signal'].diff().fillna(0)
    signals.iloc[0, signals.columns.get_loc('position_change')] = 0

    return signals['signal'], signals

def compute_bollinger_bands(prices, window_length=20, num_std=2):
    sma = moving_average(prices, window_length)
    
    # Calculate rolling std (using a simple moving window, numpy only)
    stds = np.empty_like(prices)
    half_w = window_length // 2
    
    for i in range(len(prices)):
        start = max(0, i - half_w)
        end = min(len(prices), i + half_w + 1)
        stds[i] = np.std(prices[start:end])
    
    upper_band = sma + num_std * stds
    lower_band = sma - num_std * stds
    
    return sma, upper_band, lower_band

def signal_bollinger(prices, bollinger_window_length=20, num_std=2, persistence=3):
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0

    prices_array = prices.to_numpy()
    sma, upper_band, lower_band = compute_bollinger_bands(
        prices_array, window_length=bollinger_window_length, num_std=num_std)

    # Check if prices cross back inside lower band
    outside_lower = prices_array < lower_band
    outside_lower_prev = np.roll(outside_lower, 1)
    outside_lower_prev[0] = False
    buy_signal = (outside_lower_prev == True) & (outside_lower == False)

    # Check if prices cross back inside upper band
    outside_upper = prices_array > upper_band
    outside_upper_prev = np.roll(outside_upper, 1)
    outside_upper_prev[0] = False
    sell_signal = (outside_upper_prev == False) & (outside_upper == True)

    # Initialize position logic
    position = np.zeros(len(prices_array), dtype=float)
    holding = 0
    for i in range(len(prices_array)):
        if holding == 0 and buy_signal[i]:
            holding = persistence
        if holding > 0:
            position[i] = 1.0
            holding -= 1
        if sell_signal[i]:
            holding = 0

    # Store results
    signals['signal'] = position
    signals['position_change'] = signals['signal'].diff().fillna(0)
    signals.loc[prices.index[0], 'position_change'] = 0

    return signals['signal'], signals


def signal02(price, rsi_window_length, lower_rsi_bound, upper_rsi_bound, bollinger_window_length, bollinger_n_stds, persistance):

    #RSI Signal
    rsi_sig, _ = signal_rsi(prices, rsi_window_length, lower_rsi_bound, upper_rsi_bound)

    #Bollinger Signal
    bollinger_sig, _ = signal_bollinger(prices, bollinger_window_length, bollinger_n_stds, persistence=3)

    #Combine Signals
    combined = combine_two_subsignals(rsi_sig, bollinger_sig)

    #Return combined signal dataframe
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = combined
    signals['position_change'] = signals['signal'].diff().fillna(0)

    return signals
    
#######################################################################
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


######### Optimizers ###################################################################

def gridsearch_strategy(price, param_grid, signal_fn, metric='sharpe'):
    keys = list(param_grid.keys())
    shapes = [len(param_grid[k]) for k in keys]
    total_combos = np.prod(shapes)

    all_params = []
    for i in range(total_combos):
        idxs = []
        rem = i
        for s in reversed(shapes):
            idxs.append(rem % s)
            rem //= s
        idxs = idxs[::-1]
        combo = {keys[j]: param_grid[keys[j]][idxs[j]] for j in range(len(keys))}
        all_params.append(combo)

    best_score = -np.inf
    best_params = None
    best_metrics = None
    results = []

    for param in all_params:
        signal = signal_fn(price, **param)
        cumret, sharpe_val, max_dd, volatility = backtest_strategy(price, signal)

        if metric == 'sharpe':
            score = sharpe_val
        elif metric == 'cumret':
            score = cumret
        elif metric == 'max_dd':
            score = -max_dd  # lower is better
        elif metric == 'volatility':
            score = -volatility  # lower is better
        else:
            raise ValueError("Unsupported metric.")

        results.append((param, score))

        if score > best_score:
            best_score = score
            best_params = param
            best_metrics = (cumret, sharpe_val, max_dd, volatility)

    return best_params, best_score, best_metrics, results

######################################################################################


###### Measures function ###########################

def compute_total_trading_return(df_position, initial_cash=1.0):
    final_value = df_position.iloc[-1]['stock_value'] + df_position.iloc[-1]['cash'] #Trade all leftover cash
    return (final_value - initial_cash) / initial_cash

def returns(prices, positions):
    prices_array = prices.to_numpy()
    daily_returns = (prices_array[1:] / prices_array[:-1]) - 1
    strategy_returns = positions[:-1] * daily_returns
    return strategy_returns

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

#####################################################


####### Backtesting ##############################

def backtest_strategy(price, signal):
    position = signal.shift(1).fillna(0).to_numpy()
    strat_returns = returns(price, position)
    cumret = cumulative_return(strat_returns)
    sharpe_val = sharpe(strat_returns)
    max_dd = max_drawdown(strat_returns)
    volatility = std_deviation(strat_returns)
    return cumret, sharpe_val, max_dd, volatility

##################################################


############ Benchmark functions #################

def buy_and_hold_return(df_prices):
    return (df_prices['AAPL'].iloc[-1] - df_prices['AAPL'].iloc[0]) / df_prices['AAPL'].iloc[0]


def random_trading_return(df_prices, df_price_changes):
    np.random.seed(42)
    random_signals = np.random.choice([-1, 0, 1], size=len(df_prices), p=[0.1, 0.8, 0.1])
    df_random_position_change = pd.DataFrame(data=random_signals, index=df_prices.index, columns=['position_change']) #Random position changes
    df_position = simulate_single_stock_trading(df_random_position_change, df_price_changes, df_prices, capital_fraction_per_trade=1)    
    
    return compute_total_trading_return(df_position, initial_cash=1.0)

###############################################################################

