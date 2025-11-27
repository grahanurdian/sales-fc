import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objs as go

# Set page config
st.set_page_config(page_title="Executive Sales Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- Data Loading & Preparation ---
@st.cache_data
def load_data():
    """Loads and prepares the data for the dashboard."""
    try:
        df = pd.read_csv('train.csv', encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv('train.csv')
    
    # 1. Date Parsing
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    
    # 2. Filtering (Realized Revenue only)
    df = df[df['STATUS'] == 'Shipped']
    
    # Ensure numeric
    df['SALES'] = pd.to_numeric(df['SALES'], errors='coerce')
    df['MSRP'] = pd.to_numeric(df['MSRP'], errors='coerce')
    df['PRICEEACH'] = pd.to_numeric(df['PRICEEACH'], errors='coerce')
    
    # 3. Feature Engineering
    df['sales_variance'] = df['PRICEEACH'] - df['MSRP']
    
    return df

# --- Analytics Functions ---
@st.cache_resource
def train_prophet(df):
    """Trains a Prophet model on the aggregated daily sales."""
    daily_sales = df.groupby('ORDERDATE')['SALES'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(daily_sales)
    return m

def calculate_rfm(df):
    """Calculates Recency, Frequency, and Monetary value for each customer."""
    # Reference date (usually the day after the last transaction in the dataset)
    snapshot_date = df['ORDERDATE'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CUSTOMERNAME').agg({
        'ORDERDATE': lambda x: (snapshot_date - x.max()).days, # Recency
        'ORDERNUMBER': 'nunique', # Frequency
        'SALES': 'sum' # Monetary
    }).reset_index()
    
    rfm.rename(columns={
        'ORDERDATE': 'Recency',
        'ORDERNUMBER': 'Frequency',
        'SALES': 'Monetary'
    }, inplace=True)
    
    return rfm

# --- Main App ---
def main():
    st.title("🚀 Executive Sales Dashboard")
    
    with st.spinner('Loading and processing enterprise data...'):
        df = load_data()
        
    # --- Top Level Metrics ---
    total_revenue = df['SALES'].sum()
    total_orders = df['ORDERNUMBER'].nunique()
    active_customers = df['CUSTOMERNAME'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Active Customers", f"{active_customers:,}")
    
    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Strategic Forecast", 
        "👥 Customer Segmentation", 
        "📦 Product Intelligence", 
        "🌍 Global Footprint"
    ])

    # --- Tab 1: Strategic Forecast ---
    with tab1:
        st.header("Strategic Sales Forecast")
        
        # Filter
        product_lines = df['PRODUCTLINE'].unique()
        selected_products = st.multiselect("Filter by Product Line", product_lines, default=product_lines)
        
        forecast_df = df[df['PRODUCTLINE'].isin(selected_products)]
        
        if forecast_df.empty:
            st.warning("No data for selected filters.")
        else:
            with st.spinner('Training Prophet model...'):
                m = train_prophet(forecast_df)
                
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)
            
            fig_forecast = plot_plotly(m, forecast)
            fig_forecast.update_layout(
                template='plotly_dark',
                title="Sales Forecast (1 Year Horizon)",
                xaxis_title="Date",
                yaxis_title="Sales ($)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

    # --- Tab 2: Customer Segmentation (RFM) ---
    with tab2:
        st.header("Customer Segmentation (RFM Analysis)")
        
        rfm = calculate_rfm(df)
        
        # 3D Scatter Plot
        fig_3d = px.scatter_3d(
            rfm,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Monetary',
            hover_name='CUSTOMERNAME',
            template='plotly_dark',
            title="Customer Segments: Recency vs Frequency vs Monetary",
            labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency (Orders)', 'Monetary': 'Total Spend ($)'}
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Top 10 High Value Customers
        st.subheader("🏆 Top 10 High Value Customers")
        # High Monetary, Low Recency (sorting by Monetary desc for now, could be more complex)
        top_10 = rfm.sort_values('Monetary', ascending=False).head(10)
        st.dataframe(top_10.style.format({'Monetary': "${:,.2f}"}))

    # --- Tab 3: Product Intelligence ---
    with tab3:
        st.header("Product Intelligence")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.subheader("Product Hierarchy (Sales Volume)")
            # Aggregating for Sunburst
            # Limit to top 10 products per line to avoid overcrowding
            # For simplicity in this view, we'll just take the top products overall or per group
            
            # Group by Line and Code
            prod_sales = df.groupby(['PRODUCTLINE', 'PRODUCTCODE'])['SALES'].sum().reset_index()
            
            # To make the chart readable, let's keep top 10 codes per line
            prod_sales = prod_sales.sort_values(['PRODUCTLINE', 'SALES'], ascending=[True, False])
            prod_sales = prod_sales.groupby('PRODUCTLINE').head(10)
            
            fig_sun = px.sunburst(
                prod_sales,
                path=['PRODUCTLINE', 'PRODUCTCODE'],
                values='SALES',
                template='plotly_dark',
                title="Revenue Distribution: Line > Product"
            )
            st.plotly_chart(fig_sun, use_container_width=True)
            
        with col_p2:
            st.subheader("Pricing Strategy: MSRP vs Actual")
            # Avg MSRP vs Price
            price_comp = df.groupby('PRODUCTLINE')[['MSRP', 'PRICEEACH']].mean().reset_index()
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=price_comp['PRODUCTLINE'], y=price_comp['MSRP'], name='Avg MSRP'))
            fig_bar.add_trace(go.Bar(x=price_comp['PRODUCTLINE'], y=price_comp['PRICEEACH'], name='Avg Price Each'))
            
            fig_bar.update_layout(
                barmode='group',
                template='plotly_dark',
                title="Average MSRP vs. Actual Selling Price",
                yaxis_title="Price ($)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- Tab 4: Global Footprint ---
    with tab4:
        st.header("Global Sales Footprint")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Sales by Country")
            country_sales = df.groupby('COUNTRY')['SALES'].sum().reset_index()
            
            fig_map = px.choropleth(
                country_sales,
                locations='COUNTRY',
                locationmode='country names',
                color='SALES',
                template='plotly_dark',
                title="Global Revenue Map"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        with col_g2:
            st.subheader("Sales by Territory")
            terr_sales = df.groupby('TERRITORY')['SALES'].sum().reset_index()
            
            fig_terr = px.bar(
                terr_sales,
                x='TERRITORY',
                y='SALES',
                color='SALES',
                template='plotly_dark',
                title="Revenue by Territory"
            )
            st.plotly_chart(fig_terr, use_container_width=True)

if __name__ == "__main__":
    main()
