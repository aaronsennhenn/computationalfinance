#Load libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from module import * 

#Load data in
tickers = ["TSLA", "AAPL", "AMD"]
start = "2010-01-01"
end = "2024-12-31"

df_prices, df_changes = download_stock_price_data(tickers, start, end)



def create_expanding_folds(price_series, start_date, end_date, fold_years=2):

    price_series = price_series.sort_index()
    price_series = price_series.loc[start_date:end_date]

    fold_starts = pd.date_range(start=start_date, end=end_date, freq=f'{fold_years}YS')
    folds = []

    for i in range(len(fold_starts) - 1):
        train_end = fold_starts[i]
        val_start = fold_starts[i]
        val_end = fold_starts[i + 1] - pd.Timedelta(days=1)

        train_mask = (price_series.index >= start_date) & (price_series.index < train_end)
        val_mask = (price_series.index >= val_start) & (price_series.index <= val_end)

        train_data = price_series.loc[train_mask]
        val_data = price_series.loc[val_mask]

        if not train_data.empty and not val_data.empty:
            folds.append((train_data, val_data))

    return folds



folds = create_expanding_folds(
    price_series=df_prices['AAPL'],
    start_date='2010-01-01',
    end_date='2020-12-31',
    fold_years=2
)