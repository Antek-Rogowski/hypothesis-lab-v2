import pandas as pd
import numpy as np

from dane import load_data
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



if __name__ == "__main__":
    entries = select_events(signals.mean_reversion(load_data('GLD'), 20, 2), 5)
    print(len(entries))