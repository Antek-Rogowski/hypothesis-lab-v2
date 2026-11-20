import pandas as pd
import numpy as np
from dane import load_data

def mean_reversion(df, window, k):
    # Mean_reversion -  when the price moves unusually far from its recent average,
    # it tends to come back towards it.

    #Args:
        # df: price table with a close price column and dates as index.
        # window: number of past sessions used for the rolling mean and std.
        # k: threshold in standard deviations (e.g. 2 = "two typical swings").

    #Returns:
        # Series with the same dates as df:
        # 1 -> long  (price far below its average, expect a rise),
        # -1 -> short (price far above its average, expect a fall),
        # 0 -> no signal.

    price = df["Close"]

    mean = price.rolling(window).mean()
    std_dev = price.rolling(window).std()
    z = (price - mean) / std_dev

    signal = np.select([z < -k, z > k], [1, -1], default=0)
    return pd.Series(signal, index=price.index)

def momentum(df, window, k):
    # Momentum -  when the price moves one way, it will be still moving.

    #Args:
        # df: price table with a close price column and dates as index.
        # window: number of past sessions used for the trend observation.
        # k: minimum return as a fraction (0.05 = 5%).

    #Returns:
        # Series with the same dates as df:
        # 1 -> long  (price rises, expect a rise),
        # -1 -> short (price falling, expect a fall),
        # 0 -> no signal.
    price = df["Close"]

    price_ret = price/price.shift(window) - 1
    signal = np.select([price_ret > k, price_ret < -k], [1, -1], default=0)
    return pd.Series(signal, index=price.index)

def shock_rebound(df, x_pct):
    # shock_rebound -  “after a large one-day move, the price tends to partly reverse.

    #Args:
        # df: price table with a close price column and dates as index.
        # x_pct: treshold [%].

    #Returns:
        # Series with the same dates as df:
        # 1 -> long  (price fell, expect a rise),
        # -1 -> short (price raised, expect a fall),
        # 0 -> no signal.

    price = df["Close"]
    x = x_pct/100
    diff = price/price.shift(1) - 1

    signal = np.select([diff >= x, diff <= -x], [-1, 1], default=0)
    return pd.Series(signal, index=price.index)

if __name__ == "__main__":

    df = load_data("GLD")
    check = shock_rebound(df, 3)
    print(check)

    short_count = (check == -1).sum()
    long_count = (check == 1).sum()

    print(f'Short: {short_count}')
    print(f'Long: {long_count}')

