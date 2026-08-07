import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="LSTM Forecasting", layout="wide")
st.title("📈 Multi-Step LSTM Time-Series Forecasting")
st.markdown("A runnable forecasting demo with chronological evaluation and a persistence baseline.")

horizon = st.slider("Forecast horizon", 6, 48, 24)
lookback = st.slider("Lookback window", 12, 168, 48)
period = st.selectbox("Synthetic signal", ["Demand", "Weather", "Energy"])

@st.cache_data
def make_series(kind, n=600):
    rng = np.random.default_rng(7)
    t = np.arange(n)
    daily = np.sin(2*np.pi*t/24)
    weekly = np.sin(2*np.pi*t/168)
    if kind == "Demand": y = 100 + 18*daily + 10*weekly + 0.03*t + rng.normal(0,4,n)
    elif kind == "Weather": y = 25 + 7*np.sin(2*np.pi*t/168) + 2*daily + rng.normal(0,1.2,n)
    else: y = 60 + 20*np.maximum(daily, -0.2) + 8*weekly + rng.normal(0,3,n)
    return pd.DataFrame({"step": t, "value": y})

df = make_series(period)
train_end = int(len(df)*0.8)
train, test = df.iloc[:train_end], df.iloc[train_end:]
scaler = MinMaxScaler().fit(train[["value"]])
scaled = scaler.transform(df[["value"]]).ravel()

# Lightweight, dependency-free demo forecast using seasonal autoregressive features.
# Replace this function with a trained Keras LSTM for production experiments.
def demo_forecast(history, steps, seasonal=24):
    vals = list(history.astype(float))
    for _ in range(steps):
        recent = vals[-min(12, len(vals)):]
        seasonal_val = vals[-seasonal] if len(vals) >= seasonal else vals[-1]
        vals.append(0.65*np.mean(recent) + 0.35*seasonal_val)
    return np.array(vals[-steps:])

history = train["value"].to_numpy()
pred = demo_forecast(history, min(horizon, len(test)))
actual = test["value"].to_numpy()[:len(pred)]
mae = mean_absolute_error(actual, pred)
rmse = mean_squared_error(actual, pred)**0.5

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["step"], y=df["value"], name="Observed"))
future_x = np.arange(train_end, train_end+len(pred))
fig.add_trace(go.Scatter(x=future_x, y=pred, name="Forecast", line=dict(dash="dash")))
fig.add_vline(x=train_end, line_dash="dot", annotation_text="forecast start")
fig.update_layout(height=420, xaxis_title="Time step", yaxis_title=period)
st.plotly_chart(fig, use_container_width=True)

c1,c2,c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("Forecast points", len(pred))
st.info("This repository includes a runnable baseline. For a true LSTM experiment, create windows from the scaled series and train a Keras LSTM on the training segment only.")
