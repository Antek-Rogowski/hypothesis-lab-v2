import streamlit as st

import data
import engine
import signals

st.title("Hypothesis Lab")

with st.sidebar:
    instrument = st.selectbox("Instrument", data.SYMBOLS)
    hypothesis = st.selectbox("Hypothesis", ["Mean Reversion", "Momentum", "Shock Rebound"])

    # parameters depend on the chosen hypothesis
    if hypothesis == "Mean Reversion":
        window = st.number_input("Window (sessions)", value=20, step=1, min_value=2)
        k = st.number_input("k (standard deviations)", value=2.0, step=0.1, min_value=0.0)
    elif hypothesis == "Momentum":
        window = st.number_input("Window (sessions)", value=20, step=1, min_value=1)
        k = st.number_input("k (minimum return, 0.05 = 5%)", value=0.0, step=0.01, min_value=0.0)
    elif hypothesis == "Shock Rebound":
        x_pct = st.number_input("Threshold (% one-day move)", value=3.0, step=0.5, min_value=0.1)

    horizon = st.number_input("Horizon (sessions)", value=5, step=1, min_value=1)
    run = st.button("Run", type="primary")


if run:
    df = data.load_data(instrument)

    if hypothesis == "Mean Reversion":
        sig = signals.mean_reversion(df, window, k)
    elif hypothesis == "Momentum":
        sig = signals.momentum(df, window, k)
    elif hypothesis == "Shock Rebound":
        sig = signals.shock_rebound(df, x_pct)

    res = engine.run_hypothesis(df, sig, horizon)

    if res["events"] == 0:
        st.info("The hypothesis produced no events with these settings.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Events", res["events"])
        col2.metric("Hit rate", f"{res['hit_rate']:.1%}", f"{res['diff'] * 100:+.1f} pp vs baseline",
                    delta_color="off")
        col3.metric("Baseline", f"{res['baseline']:.1%}")
        col4.metric("p-value", f"{res['p_value']:.3f}")

        if res["p_value"] < 0.05 and res["diff"] > 0:
            st.success("Significantly better than random entries (p < 0.05).")
        elif res["p_value"] < 0.05 and res["diff"] < 0:
            st.warning("Significantly worse than random entries (p < 0.05).")
        else:
            st.info("No evidence that the rule differs from random entries (p ≥ 0.05).")