# Define parameters
short_window = 50
long_window = 100

# Use AAPL data
ticker = 'AAPL'
price_series = df_prices[ticker]
price_changes = df_price_changes[ticker]

# --- First Code (Simplified Logic) ---

def backtest_ma(price_series, short_window, long_window):
    signals = ma_signal(price_series, short_window, long_window)
    position = signals['signal'].fillna(0).to_numpy()
    prices_array = price_series.to_numpy()
    daily_returns = (prices_array[1:] / prices_array[:-1]) - 1
    strategy_returns = position[:-1] * daily_returns
    
    cumret = np.prod(1 + strategy_returns) - 1
    return cumret, strategy_returns

simple_cumret, simple_returns = backtest_ma(price, short_window, long_window)


# --- Second Code (Realistic Portfolio Simulation) ---

signals_detailed = ma_signal(price, short_window, long_window)
position_changes = signals_detailed['position_change'].to_frame()

df_position_detailed = simulate_single_stock_trading(position_changes, price_change, price, capital_fraction_per_trade=1)
detailed_cumret = compute_total_trading_return(df_position_detailed)

# --- Print Comparison ---
print(f"Simplified Strategy Cumulative Return: {simple_cumret:.4f}")
print(f"Realistic Strategy Cumulative Return:  {detailed_cumret:.4f}")