import sys
import os
sys.path.append(r'C:\Users\aaron\OneDrive\Desktop\Data Science Studium\SS 25\Computational Python')

from module import *

# Download the data ###########################################################

tickers = [ \
    'AAPL', # Apple
    'MSFT', # Microsoft
    'AMZN', # Amazon
    '^GSPC'] # S&P500 - Benchmark

# define time span of stock price data
start_date = '2010-01-01'
end_date = '2021-12-31'
    
# download the data
df_prices, df_price_changes = download_stock_price_data(tickers, start_date, end_date)

###############################################################################


short_range = range(10, 100, 10)
long_range = range(25, 250, 25) 


#Run Gridsearch
best_params, best_ret, best_position, combinations_df  = optimize_ma_params(df_prices['AAPL'], df_price_changes['AAPL'], short_range, long_range)


#Benchmark the result
buy_and_hold_return = buy_and_hold_return(df_prices)
sanity_check_ret = random_trading_return(df_prices['AAPL'], df_price_changes['AAPL'])
