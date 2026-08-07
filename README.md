# 📈 Time-Series Forecasting with LSTM

A multi-step forecasting project for demand, weather, energy, or financial time series. It compares a persistence baseline with an LSTM sequence model and visualizes forecast uncertainty and error metrics.

## How it works
1. Sort observations chronologically and resample to a consistent frequency.
2. Fill missing values and scale the target using training data only.
3. Create sliding windows: the previous `lookback` observations become model input and the next `horizon` values become targets.
4. Train an LSTM encoder that learns temporal dependencies and a dense decoder that predicts multiple future points.
5. Evaluate with a chronological holdout using MAE, RMSE, and MAPE.
6. In production, retrain periodically and monitor drift, seasonality, and forecast error.

## Tech stack
- Python, NumPy, Pandas
- TensorFlow/Keras LSTM
- Scikit-learn preprocessing and metrics
- Plotly, Streamlit

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interview talking points
1. **Chronological splits prevent leakage:** random train/test splitting lets future patterns leak into training and produces unrealistic metrics.
2. **Multi-step direct forecasting is practical:** the model predicts the complete horizon in one pass, avoiding recursive error accumulation from repeatedly feeding predictions back as inputs.
3. **A naive baseline is mandatory:** if the LSTM cannot beat “tomorrow equals today” or seasonal-naive forecasting, the added complexity is not justified.
