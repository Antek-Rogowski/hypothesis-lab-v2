import pandas as pd
import numpy as np
from data import load_data

# Mean_reversion -  when the price moves unusually far from its recent average, then it should come back to it
def mean_reversion(df, window, k):
    #Args:
        # df:       price table with a close price column and dates as index
        # window:   number of past sessions used for the rolling mean and std
        # k:        threshold in standard deviations (e.g. 2 = "two typical swings")

    #Returns:
        # Series with the same dates and signals
        #  1    -> long  (price below, expect rise)
        # -1    -> short (price above, expect fall)
        #  0    -> no signal

    price = df["Close"]

    # Statistics
    mean = price.rolling(window).mean()
    std_dev = price.rolling(window).std()
    z = (price - mean) / std_dev

    signal = np.select([z < -k, z > k], [1, -1], default=0)
    return pd.Series(signal, index=price.index)

# Momentum -  when the price moves one way, it will be still moving
def momentum(df, window, k):
    #Args:
        # df:       price table with a close price column and dates as index
        # window:   number of past sessions used for the trend observation
        # k:        minimum return as a fraction (0.05 = 5%)

    #Returns:
        # Series with the same dates and signals
        #  1    ->  long  (price rises, expect rise)
        # -1    -> short (price falling, expect fall)
        #  0    -> no signal

    price = df["Close"]

    # Current price / price from {window} past days 
    price_ret = price/price.shift(window) - 1
    signal = np.select([price_ret > k, price_ret < -k], [1, -1], default=0)
    return pd.Series(signal, index=price.index)

# Shock_rebound -  after a huge one-day move, the price tends to reverse
def shock_rebound(df, x_pct):
    #Args:
        # df: price table with a close price column and dates as index
        # x_pct: treshold [%]

    #Returns:
        # Series with the same dates and signals
        #  1    -> long  (price fell, expect rise)
        # -1    -> short (price raised, expect fall)
        #  0    -> no signal

    price = df["Close"]
    x = x_pct/100
    diff = price/price.shift(1) - 1

    signal = np.select([diff >= x, diff <= -x], [-1, 1], default=0)
    return pd.Series(signal, index=price.index)

if __name__ == "__main__":

    # Test run
    df = load_data("GLD")
    check = momentum(df, 1, 3)
    print(check)

    short_count = (check == -1).sum()
    long_count = (check == 1).sum()

    print(f'Short: {short_count}')
    print(f'Long: {long_count}')