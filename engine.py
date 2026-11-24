# signals -> hyphotesis grade

import pandas as pd
import numpy as np
# p-value test
from scipy.stats import binomtest

from data import load_data
import signals

# Event selection
def select_events(sig, horizon):
    print(sig)
    values = sig.to_numpy()
    result = []
    earliest_possible = 0

    print(values)
    # Filtering raised flags into valid signals (3 days horizon, 1 type of signal at time, edge dates)
    # at this stage only flags, without dates
    for i in range(len(values)):
        if values[i] != 0 and i >= earliest_possible and i + horizon <= len(values) - 1:
            result.append(i)              
            earliest_possible = i + horizon
    # idx of selected flags
    return result

# Builds a table of events
def build_trades(df, sig, horizon):
    # Imports: list of selected events idx, list of prices, list of signals and combines it into one list of dicts
    prices = df["Close"].to_numpy()
    sides = sig.to_numpy()
    rows = []

    for i in select_events(sig, horizon):
        side = sides[i]                     # 1 = long, -1 = short
        entry_price = prices[i]             # Close on the signal day
        exit_price = prices[i + horizon]    # Close `horizon` sessions later

        # long: exit/entry - 1, short: 1 - exit/entry = -(exit/entry - 1)
        ret = side * (exit_price / entry_price - 1)

        rows.append({
            "signal_date": df.index[i],
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": ret,
            "hit": ret > 0,
        })
    # list of dicts -> table
    return pd.DataFrame(rows)

# Hit rate if entered on random date with the same mix of long/short as a rule
def baseline(df, trades, horizon):
    price = df["Close"]
    # future return over `horizon` sessions
    ret = price.shift(-horizon) / price - 1
    # drop of NaaN     
    ret = ret.dropna()                          

    p_long = (ret > 0).mean()                   # share of days when long would win
    p_short = (ret < 0).mean()                  # share of days when short would win

    n_long = (trades["side"] == 1).sum()        # how many longs the rule made, calculated only on selected signals
    n_short = (trades["side"] == -1).sum()      # how many shorts the rule made, calclulated only on selected signals

    # weighted average: same long/short mix as the rule
    return (n_long * p_long + n_short * p_short) / (n_long + n_short)                     

# Measuremnt how likely result is to be random
def p_value(trades, base):
    k = trades["hit"].sum()                         # number of hits (True counts as 1)
    n = len(trades)                                 # number of events (rows in trades)
    return binomtest(k=k, n=n, p=base).pvalue       # p < 0,05 => rule not random

# Full evaluation of one hypothesis
def run_hypothesis(df, sig, horizon):

    trades = build_trades(df, sig, horizon)                 
    if len(trades) == 0:                    
        return {"events": 0}

    base = baseline(df, trades, horizon)
    hit_rate = trades["hit"].mean()
    p = p_value(trades, base)               

    # Always return dict, even when 0 events, easy display in app
    return {
        "events": len(trades),
        "n_long": (trades["side"] == 1).sum(),
        "n_short": (trades["side"] == -1).sum(),
        "hits": trades["hit"].sum(),
        "hit_rate": hit_rate,
        "baseline": base,
        "diff": hit_rate - base,
        "p_value": p,
        "trades": trades,
    }


if __name__ == "__main__":
    # Test run
    df = load_data("GLD")
    result = run_hypothesis(df, signals.mean_reversion(df, 20, 2), 5)
    for key, value in result.items():
        if key != "trades":
            print(f"{key}: {value}")

    print(run_hypothesis(df, signals.mean_reversion(df, 20, 10), 5))