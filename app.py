import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(page_title="Retail Sales Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Data Loading & Engineering ---
@st.cache_data
def load_data():
    """Loads and cleans the training data."""
    try:
        df = pd.read_csv('train.csv', encoding='ISO-8859-1') # Common encoding for this dataset type
    except UnicodeDecodeError:
        df = pd.read_csv('train.csv')
    
    # Date parsing
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    
    # Ensure numeric
    df['SALES'] = pd.to_numeric(df['SALES'], errors='coerce')
    
    return df

@st.cache_data
def perform_segmentation(df):
    """Performs K-Means clustering on Customers."""
    # Group by Customer
    customer_df = df.groupby('CUSTOMERNAME').agg({
        'SALES': 'sum',
        'ORDERNUMBER': 'nunique' # Transaction count
    }).reset_index()
    
    # Simple clustering on Sales and Frequency
    X = customer_df[['SALES', 'ORDERNUMBER']]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_df['Cluster'] = kmeans.fit_predict(X)
    
    # Label clusters based on average Sales
    cluster_avg = customer_df.groupby('Cluster')['SALES'].mean().sort_values()
    
    # Map cluster IDs to labels: 0 -> Low, 1 -> Steady, 2 -> High (based on sort order)
    label_map = {}
    labels = ["Low Volume", "Steady Average", "High Performers"]
    for i, cluster_id in enumerate(cluster_avg.index):
        label_map[cluster_id] = labels[i]
        
    customer_df['Segment'] = customer_df['Cluster'].map(label_map)
    
    # Merge Segment back to main dataframe
    merged_df = df.merge(customer_df[['CUSTOMERNAME', 'Segment']], on='CUSTOMERNAME', how='left')
    
    return merged_df, customer_df

# --- Model Training ---
@st.cache_resource
def train_prophet(df):
    """Trains a Prophet model on the aggregated daily sales."""
    # Aggregate to daily
    daily_sales = df.groupby('ORDERDATE')['SALES'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(daily_sales)
    return m

# --- Main App ---
def main():
    st.title("📊 Retail Sales Intelligence Dashboard")
    
    # Load Data
    with st.spinner('Loading and analyzing data...'):
        raw_df = load_data()
        df, customer_stats = perform_segmentation(raw_df)

    # Tabs
    tab1, tab2 = st.tabs(["🏢 Business Insights", "📈 Forecast Engine"])

    # --- Tab 1: Business Insights ---
    with tab1:
        st.header("Customer Segmentation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales by Segment")
            # Aggregate sales by segment
            segment_sales = df.groupby('Segment')['SALES'].sum().reset_index()
            
            fig_bar = px.bar(
                segment_sales, 
                x='Segment', 
                y='SALES', 
                color='Segment',
                template='plotly_dark',
                title="Total Sales Volume by Customer Segment",
                labels={'SALES': 'Total Sales ($)'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.subheader("Cluster Distribution")
            fig_scatter = px.scatter(
                customer_stats,
                x='ORDERNUMBER',
                y='SALES',
                color='Segment',
                template='plotly_dark',
                title="Customer Value: Sales vs. Transaction Count",
                labels={'SALES': 'Total Lifetime Sales ($)', 'ORDERNUMBER': 'Number of Orders'},
                hover_data=['CUSTOMERNAME']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        st.subheader("Data Preview")
        st.dataframe(df.head())

    # --- Tab 2: Forecast Engine ---
    with tab2:
        st.header("Sales Forecasting")
        
        # Sidebar-like filters within the tab (or use st.sidebar)
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            selected_segments = st.multiselect(
                "Filter by Customer Segment",
                options=df['Segment'].unique(),
                default=df['Segment'].unique()
            )
            
        with col_filter2:
            selected_products = st.multiselect(
                "Filter by Product Family",
                options=df['PRODUCTLINE'].unique(),
                default=df['PRODUCTLINE'].unique()
            )
            
        # Filter Logic
        filtered_df = df[
            (df['Segment'].isin(selected_segments)) & 
            (df['PRODUCTLINE'].isin(selected_products))
        ]
        
        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
        else:
            # Train Model
            with st.spinner('Training forecast model on filtered data...'):
                m = train_prophet(filtered_df)
                
            # Forecast Horizon
            days_to_forecast = st.slider("Forecast Horizon (Days)", 30, 365, 90)
            
            # Predict
            future = m.make_future_dataframe(periods=days_to_forecast)
            forecast = m.predict(future)
            
            # Visualize
            st.subheader(f"Sales Forecast ({days_to_forecast} days)")
            
            # Custom Plotly chart for better aesthetics
            fig_forecast = plot_plotly(m, forecast)
            fig_forecast.update_layout(
                template='plotly_dark',
                title="Projected Sales Trend",
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Components
            with st.expander("View Forecast Components"):
                fig_comp = m.plot_components(forecast)
                st.write(fig_comp)

if __name__ == "__main__":
    main()
