import pandas as pd
import pytest

from engine import select_events, build_trades, baseline, p_value, run_hypothesis


def exercise_df():
    # test data
    return pd.DataFrame({
        "Close": [100, 98, 97, 99, 101, 100, 96, 95, 97, 99],
        "Signal": [0, 1, 1, 0, -1, 0, 1, 0, 1, 0],
    })


# --- select_events ----------------------------------------------------------

def test_select_events_basic():
    # day 1 long (exit day 4), day 2 skipped (position open),
    # day 4 short (exit day 7), day 6 skipped, day 8 exit would be day 11 -> no data
    df = exercise_df()
    assert select_events(df["Signal"], 3) == [1, 4]


def test_select_events_exit_must_fit():
    # only 7 sessions (0-6): the short from day 4 would exit on day 7 -> outside the data
    df = exercise_df().iloc[:7]
    assert select_events(df["Signal"], 3) == [1]


def test_select_events_horizon_1():
    # horizon 1: every position is closed the next day, so every signal day fits
    df = exercise_df()
    assert select_events(df["Signal"], 1) == [1, 2, 4, 6, 8]


# --- build_trades -----------------------------------------------------------

def test_trade_returns():
    df = exercise_df()
    trades = build_trades(df, df["Signal"], 3)

    assert len(trades) == 2
    assert trades["side"].tolist() == [1, -1]
    assert trades["entry_price"].tolist() == [98, 101]
    assert trades["exit_price"].tolist() == [101, 95]
    # long: 101/98 - 1 = +3.06%, short: 1 - 95/101 = +5.94%
    assert trades["return"].tolist() == pytest.approx([0.0306, 0.0594], abs=1e-4)
    assert trades["hit"].tolist() == [True, True]


# --- baseline and p-value ---------------------------------------------------

def test_baseline():
    # 7 days with a 3-session future return: 3 up, 4 down
    # one long and one short -> (1 * 3/7 + 1 * 4/7) / 2 = 0.5
    df = exercise_df()
    trades = build_trades(df, df["Signal"], 3)
    assert baseline(df, trades, 3) == pytest.approx(0.5)


def test_p_value():
    # 2 hits out of 2 with p = 0.5: P(2 hits) + P(0 hits) = 0.25 + 0.25
    df = exercise_df()
    trades = build_trades(df, df["Signal"], 3)
    assert p_value(trades, 0.5) == pytest.approx(0.5)


# --- run_hypothesis ---------------------------------------------------------

def test_run_hypothesis_full():
    df = exercise_df()
    result = run_hypothesis(df, df["Signal"], 3)

    assert result["events"] == 2
    assert result["n_long"] == 1
    assert result["n_short"] == 1
    assert result["hits"] == 2
    assert result["hit_rate"] == pytest.approx(1.0)
    assert result["baseline"] == pytest.approx(0.5)
    assert result["diff"] == pytest.approx(0.5)
    assert result["p_value"] == pytest.approx(0.5)


def test_no_events():
    # a signal of only zeros must not crash: the result is just {"events": 0}
    df = exercise_df()
    no_signal = pd.Series(0, index=df.index)
    assert run_hypothesis(df, no_signal, 3) == {"events": 0}
