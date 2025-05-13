import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA

# Configure Streamlit page
st.set_page_config(
    page_title="📈 Stock Forecast App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for user input
with st.sidebar:
    st.title("🔧 Settings")
    stock = st.text_input("Enter Stock Symbol", "AAPL")
    start_date = st.date_input("Start Date", date(2022, 1, 1))
    end_date = st.date_input("End Date", date.today())
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=60, value=30)

# Fetch stock data
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True)
    return data

df = load_data(stock, start_date, end_date)

# Show latest data
st.markdown("## 📊 Latest Data")
st.dataframe(df.tail(), use_container_width=True)

# Plot historical close prices
st.markdown("## 📈 Historical Price")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
fig.update_layout(title=f"{stock} Closing Prices", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# Forecast future prices
st.markdown(f"## 🔮 {forecast_days}-Day Forecast")

df.set_index("Date", inplace=True)
model = ARIMA(df["Close"], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=forecast_days)

forecast_index = pd.date_range(df.index[-1] + timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_index)

# Plot forecast
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Historical"))
fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Forecast"], name="Forecast"))
fig2.update_layout(title="Stock Price Forecast", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig2, use_container_width=True)

# Download button
csv = forecast_df.to_csv().encode("utf-8")
st.download_button("📥 Download Forecast CSV", csv, f"{stock}_forecast.csv", "text/csv")