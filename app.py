import altair as alt
import pandas as pd
import streamlit as st

import data
import engine
import signals

st.set_page_config(page_title="Hypothesis Lab", layout="wide")
st.title("Hypothesis Lab")
st.caption("Test simple trading rules on real daily data: hit rate vs. random entries, with a p-value.")

# session_state survives reruns of the script (normal variables do not)
if "history" not in st.session_state:
    st.session_state["history"] = []      # list of distinct settings in this session
if "result" not in st.session_state:
    st.session_state["result"] = None     # last computed result, shown until the next Run


@st.cache_data(ttl=3600)
def load(symbol):
    return data.load_data(symbol)


# --- Sidebar: choices ---------------------------------------------------------
with st.sidebar:
    instrument = st.selectbox("Instrument", data.SYMBOLS)
    hypothesis = st.selectbox("Hypothesis", ["Mean Reversion", "Momentum", "Shock Rebound"])

    
    if hypothesis == "Mean Reversion":
        st.caption("Long when the price is k standard deviations below its W-day mean, short when above.")
        window = st.number_input("Window (sessions)", value=20, step=1, min_value=2)
        k = st.number_input("k (standard deviations)", value=2.0, step=0.1, min_value=0.0)
        params = {"window": window, "k": k}
    elif hypothesis == "Momentum":
        st.caption("Long after a rise over the last W sessions, short after a fall (beyond threshold k).")
        window = st.number_input("Window (sessions)", value=20, step=1, min_value=1)
        k = st.number_input("k (minimum return, 0.05 = 5%)", value=0.0, step=0.01, min_value=0.0)
        params = {"window": window, "k": k}
    else:
        st.caption("Long after a one-day fall of at least X%, short after a one-day rise of at least X%.")
        x_pct = st.number_input("Threshold (% one-day move)", value=3.0, step=0.5, min_value=0.1)
        params = {"x_pct": x_pct}

    horizon = st.number_input("Horizon (sessions)", value=5, step=1, min_value=1)
    run = st.button("Run", type="primary")


config = {"instrument": instrument, "hypothesis": hypothesis, **params, "horizon": horizon}

# --- After clicking: compute and store ----------------------------------------
if run:
    df = load(instrument)

    if hypothesis == "Mean Reversion":
        sig = signals.mean_reversion(df, window, k)
    elif hypothesis == "Momentum":
        sig = signals.momentum(df, window, k)
    else:
        sig = signals.shock_rebound(df, x_pct)

    res = engine.run_hypothesis(df, sig, horizon)

    # trial counter
    if config not in st.session_state["history"]:
        st.session_state["history"].append(config)

    st.session_state["result"] = {"res": res, "config": config, "prices": df["Close"]}

# --- Show the last result -----------------------------------------------------
stored = st.session_state["result"]

if stored is None:
    st.info("Choose an instrument, a hypothesis and its parameters in the sidebar, then click Run.")
    st.stop()                              

res, cfg, prices = stored["res"], stored["config"], stored["prices"]

st.subheader(f"{cfg['instrument']}: {cfg['hypothesis']}")
st.caption(f"Data: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} sessions) | "
           f"settings: {', '.join(f'{key}={value}' for key, value in cfg.items() if key not in ('instrument', 'hypothesis'))}")

if cfg != config:
    st.info("The sidebar settings differ from the result shown. Click Run to apply them.")

if res["events"] == 0:
    st.info("The hypothesis produced no events with these settings.")
    st.stop()

# multiple testing: each extra trial raises the chance of a lucky result,
# so the p-value is multiplied by the number of trials (Bonferroni correction)
trials = len(st.session_state["history"])
p_adj = min(1.0, res["p_value"] * trials)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Events", res["events"])
col1.caption(f"long {res['n_long']}, short {res['n_short']}")
col2.metric("Hit rate", f"{res['hit_rate']:.1%}", f"{res['diff'] * 100:+.1f} pp vs baseline", delta_color="off")
col3.metric("Baseline", f"{res['baseline']:.1%}")
col3.caption("same long/short mix entered on every day")
col4.metric("p-value", f"{res['p_value']:.3f}")
col4.caption(f"after correction for {trials} trial(s): {p_adj:.3f}")

if p_adj < 0.05 and res["diff"] > 0:
    st.success("Significantly better than random entries (corrected p < 0.05).")
elif p_adj < 0.05 and res["diff"] < 0:
    st.warning("Significantly worse than random entries (corrected p < 0.05).")
else:
    st.info("No evidence that the rule differs from random entries (corrected p ≥ 0.05).")

if res["events"] < 30:
    st.warning(f"Only {res['events']} events: the result is very uncertain.")

tab_trades, tab_chart, tab_history = st.tabs(["Events", "Chart", "Trial history"])

with tab_trades:
    trades = res["trades"].copy()
    trades["side"] = trades["side"].map({1: "long", -1: "short"})
    st.dataframe(trades, hide_index=True)
    st.download_button("Download events (CSV)", trades.to_csv(index=False).encode("utf-8"),
                       file_name=f"events_{cfg['instrument']}.csv", mime="text/csv")

with tab_chart:
    price_df = prices.rename("price").reset_index()           # columns: Date, price
    line = alt.Chart(price_df).mark_line().encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("price:Q", title="Adjusted close", scale=alt.Scale(zero=False)),
    )
    points_df = res["trades"].assign(
        result=res["trades"]["hit"].map({True: "hit", False: "miss"}),
        direction=res["trades"]["side"].map({1: "long", -1: "short"}),
    )
    points = alt.Chart(points_df).mark_point(filled=True, size=50).encode(
        x="signal_date:T",
        y="entry_price:Q",
        color=alt.Color("result:N", scale=alt.Scale(domain=["hit", "miss"], range=["#2e7d32", "#c62828"])),
        shape="direction:N",
        tooltip=["signal_date:T", "direction:N", "entry_price:Q", "exit_price:Q", "return:Q", "result:N"],
    )
    st.altair_chart(line + points)
    st.caption("Each point is an entry: green = hit, red = miss; shape = long or short.")

with tab_history:
    st.caption(f"{trials} distinct trial(s) in this session. The list is cleared when the page is refreshed.")
    st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True)