import pandas as pd
import numpy as np
from scipy.stats import binomtest

from data import load_data
import signals

# Event selection
def select_events(sig, horizon):
    values = sig.to_numpy()
    result = []
    earliest_possible = 0

    # One session a time, with signal raised and horizon within the scope
    for i in range(len(values)):
        if values[i] != 0 and i >= earliest_possible and i + horizon <= len(values) - 1:
            result.append(i)              
            earliest_possible = i + horizon
    return result

# Builds a table of events
def build_trades(df, sig, horizon):
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

    return pd.DataFrame(rows)

# Baseline: hit rate of the same long/short mix if we entered on every possible day
def baseline(df, trades, horizon):
    price = df["Close"]
    ret = price.shift(-horizon) / price - 1     # future return over `horizon` sessions
    ret = ret.dropna()                          # last `horizon` days have no future price

    p_long = (ret > 0).mean()                   # share of days after which a long would win
    p_short = (ret < 0).mean()                  # share of days after which a short would win

    n_long = (trades["side"] == 1).sum()        # how many longs the rule made
    n_short = (trades["side"] == -1).sum()      # how many shorts the rule made

    # weighted average: same long/short mix as the rule
    return (n_long * p_long + n_short * p_short) / (n_long + n_short)                     

# Measuremnt how likely result is to be a random thing
def p_value(trades, base):
    k = trades["hit"].sum()                         # number of hits (True counts as 1)
    n = len(trades)                                 # number of events (rows in trades)
    return binomtest(k=k, n=n, p=base).pvalue       # p < 0,05 => rule is better than random

# Full evaluation of one hypothesis: events, hit rate, baseline, p-value
def run_hypothesis(df, sig, horizon):
    trades = build_trades(df, sig, horizon)
                     
    if len(trades) == 0:                    # No events: return the same type (dict), so the app can check result["events"] == 0
        return {"events": 0}

    base = baseline(df, trades, horizon)
    hit_rate = trades["hit"].mean()
    p = p_value(trades, base)               

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
    df = load_data("GLD")

    result = run_hypothesis(df, signals.mean_reversion(df, 20, 2), 5)
    for key, value in result.items():
        if key != "trades":
            print(f"{key}: {value}")

    print(run_hypothesis(df, signals.mean_reversion(df, 20, 10), 5))