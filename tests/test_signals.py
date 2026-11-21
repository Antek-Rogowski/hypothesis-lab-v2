import numpy as np
import pandas as pd
import pytest

from signals import mean_reversion, momentum, shock_rebound


def prices(values):
    """Small price table with only the "Close" column."""
    return pd.DataFrame({"Close": values})


def test_mean_reversion_small_example():
    # window 3: z on day 3 = (95 - 99.33) / 3.79 = -1.14 < -1 -> long
    # days 0-1 have no full window (NaN) -> 0
    df = prices([100, 102, 101, 95, 99])
    assert mean_reversion(df, 3, 1).tolist() == [0, 0, 0, 1, 0]


def test_mean_reversion_higher_threshold():
    # same data, k = 2: z = -1.14 is not below -2 -> no signal at all
    df = prices([100, 102, 101, 95, 99])
    assert mean_reversion(df, 3, 2).tolist() == [0, 0, 0, 0, 0]


def test_momentum_small_example():
    # 1-session returns: +4% -> long, -3.85% -> short, -4% -> short
    df = prices([100, 104, 100, 96])
    assert momentum(df, 1, 0.02).tolist() == [0, 1, -1, -1]


def test_shock_rebound_small_example():
    # one-day moves of at least 3%: rise -> short, fall -> long (opposite to momentum)
    df = prices([100, 104, 100, 96])
    assert shock_rebound(df, 3).tolist() == [0, -1, 1, 1]


def test_shock_rebound_below_threshold():
    # a 2% move is below the 3% threshold -> no signal
    df = prices([100, 102, 100])
    assert shock_rebound(df, 3).tolist() == [0, 0, 0]


# --- no look-ahead ----------------------------------------------------------

def random_prices(n=300, seed=1):
    """Random walk prices: reproducible thanks to the fixed seed."""
    rng = np.random.default_rng(seed)
    return prices(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))


@pytest.mark.parametrize("make_signal", [
    lambda df: mean_reversion(df, 20, 1),
    lambda df: momentum(df, 10, 0.0),
    lambda df: shock_rebound(df, 1),
])
def test_signals_do_not_look_into_the_future(make_signal):
    # change the last 5 prices: signals up to that point must stay exactly the same
    df = random_prices()
    changed = df.copy()
    changed.iloc[-5:, 0] *= 2

    before = make_signal(df).iloc[:-5]
    after = make_signal(changed).iloc[:-5]
    assert before.tolist() == after.tolist()
