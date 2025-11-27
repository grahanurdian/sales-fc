import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Set page config
st.set_page_config(page_title="Retail Sales Forecast", layout="wide")

@st.cache_data
def generate_dummy_data():
    """Generates 2 years of daily retail sales data with seasonality and trend."""
    dates = pd.date_range(start='2023-01-01', periods=365*2, freq='D')
    
    # Trend
    trend = np.linspace(0, 100, len(dates))
    
    # Seasonality (Weekly)
    # 0=Monday, 6=Sunday. Let's make weekends higher.
    seasonality = np.array([10 if d.weekday() >= 5 else -5 for d in dates])
    
    # Noise
    noise = np.random.normal(0, 5, len(dates))
    
    # Base value
    base = 500
    
    y = base + trend + seasonality + noise
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    return df

@st.cache_resource
def train_model(df):
    """Trains a Prophet model on the provided dataframe."""
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df)
    return m

def main():
    st.title("Retail Sales Forecasting with Prophet")
    st.markdown("""
    This application demonstrates time series forecasting using Facebook Prophet.
    The data is synthetically generated to simulate retail sales with a weekly pattern and upward trend.
    """)

    # 1. Load and Display Data
    with st.spinner('Generating data...'):
        df = generate_dummy_data()
    
    st.subheader("Historical Data")
    st.dataframe(df.tail())

    # 2. Train Model
    with st.spinner('Training model...'):
        m = train_model(df)

    # 3. Forecast Configuration
    st.sidebar.header("Forecast Configuration")
    days_to_forecast = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=365, value=90)

    # 4. Make Prediction
    future = m.make_future_dataframe(periods=days_to_forecast)
    forecast = m.predict(future)

    # 5. Visualization
    st.subheader("Forecast Visualization")
    
    # Plotly chart
    fig = plot_plotly(m, forecast)
    fig.update_layout(
        title=f"Sales Forecast ({days_to_forecast} days ahead)",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show forecast data
    st.subheader("Forecast Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Download button
    csv = forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Forecast CSV",
        csv,
        "forecast.csv",
        "text/csv",
        key='download-csv'
    )

if __name__ == "__main__":
    main()
