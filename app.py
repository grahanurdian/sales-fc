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
        df = pd.read_csv('train.csv', encoding='ISO-8859-1')
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
        'ORDERNUMBER': 'nunique'
    }).reset_index()
    
    # Simple clustering on Sales and Frequency
    X = customer_df[['SALES', 'ORDERNUMBER']]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_df['Cluster'] = kmeans.fit_predict(X)
    
    # Label clusters based on average Sales
    cluster_avg = customer_df.groupby('Cluster')['SALES'].mean().sort_values()
    
    # Map cluster IDs to labels: 0 -> Low, 1 -> Steady, 2 -> High
    label_map = {}
    labels = ["Low Volume", "Steady Average", "High Performers"]
    for i, cluster_id in enumerate(cluster_avg.index):
        label_map[cluster_id] = labels[i]
        
    customer_df['Segment'] = customer_df['Cluster'].map(label_map)
    
    # Merge Segment back to main dataframe
    merged_df = df.merge(customer_df[['CUSTOMERNAME', 'Segment']], on='CUSTOMERNAME', how='left')
    
    return merged_df, customer_df

# --- Deep Dive Analytics Functions ---
def perform_pareto(df):
    """Calculates and plots Pareto analysis for Product Lines."""
    # Group by Product Line
    pareto_df = df.groupby('PRODUCTLINE')['SALES'].sum().reset_index()
    pareto_df = pareto_df.sort_values('SALES', ascending=False)
    
    # Calculate cumulative percentage
    pareto_df['cumulative_sales'] = pareto_df['SALES'].cumsum()
    pareto_df['cumulative_percent'] = pareto_df['cumulative_sales'] / pareto_df['SALES'].sum() * 100
    
    # Generate Insight Text
    # Find how many products make up 80% of revenue
    top_products = pareto_df[pareto_df['cumulative_percent'] <= 80]
    num_top = len(top_products)
    if num_top == 0: # In case the first one is already > 80%
        num_top = 1
        
    insight_text = f"Top {num_top} product families generate ~{int(pareto_df.iloc[num_top-1]['cumulative_percent'])}% of total revenue."
    
    # Plot Combo Chart
    fig = go.Figure()
    
    # Bar chart for Sales
    fig.add_trace(go.Bar(
        x=pareto_df['PRODUCTLINE'],
        y=pareto_df['SALES'],
        name='Sales',
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Line chart for Cumulative %
    fig.add_trace(go.Scatter(
        x=pareto_df['PRODUCTLINE'],
        y=pareto_df['cumulative_percent'],
        name='Cumulative %',
        yaxis='y2',
        mode='lines+markers',
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.update_layout(
        title='Pareto Analysis: Revenue Concentration by Product Line',
        xaxis_title='Product Line',
        yaxis=dict(title='Total Sales ($)'),
        yaxis2=dict(
            title='Cumulative Percentage (%)',
            overlaying='y',
            side='right',
            range=[0, 110]
        ),
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(x=0.8, y=0.9)
    )
    
    return fig, insight_text

def plot_heatmap(df):
    """Plots a heatmap of Average Sales by Month and Day of Week."""
    # Extract time features
    df['Month'] = df['ORDERDATE'].dt.month_name()
    df['DayOfWeek'] = df['ORDERDATE'].dt.day_name()
    
    # Order for plotting
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Pivot table
    heatmap_data = df.pivot_table(
        index='Month', 
        columns='DayOfWeek', 
        values='SALES', 
        aggfunc='mean'
    )
    
    # Reindex to ensure correct order
    heatmap_data = heatmap_data.reindex(index=month_order, columns=day_order)
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Day of Week", y="Month", color="Avg Sales ($)"),
        x=day_order,
        y=month_order,
        color_continuous_scale='Viridis',
        title="Weekly Sales Intensity (Average Sales Heatmap)",
        template='plotly_dark'
    )
    fig.update_xaxes(side="top")
    
    return fig

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
    tab1, tab2, tab3 = st.tabs(["🏢 Business Insights", "📈 Forecast Engine", "🔍 Deep Dive Analytics"])

    # --- Tab 1: Business Insights ---
    with tab1:
        st.header("Customer Segmentation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales by Segment")
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
            
        filtered_df = df[
            (df['Segment'].isin(selected_segments)) & 
            (df['PRODUCTLINE'].isin(selected_products))
        ]
        
        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
        else:
            with st.spinner('Training forecast model on filtered data...'):
                m = train_prophet(filtered_df)
                
            days_to_forecast = st.slider("Forecast Horizon (Days)", 30, 365, 90)
            
            future = m.make_future_dataframe(periods=days_to_forecast)
            forecast = m.predict(future)
            
            st.subheader(f"Sales Forecast ({days_to_forecast} days)")
            
            fig_forecast = plot_plotly(m, forecast)
            fig_forecast.update_layout(
                template='plotly_dark',
                title="Projected Sales Trend",
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            with st.expander("View Forecast Components"):
                fig_comp = m.plot_components(forecast)
                st.write(fig_comp)

    # --- Tab 3: Deep Dive Analytics ---
    with tab3:
        st.header("Deep Dive Analytics")
        
        # 1. Pareto Analysis
        st.subheader("Revenue Concentration (Pareto)")
        fig_pareto, insight = perform_pareto(df)
        st.info(f"💡 **Insight**: {insight}")
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        st.markdown("---")
        
        # 2. Time-Series Heatmap
        st.subheader("Weekly Sales Intensity")
        st.markdown("Average sales performance by Month and Day of the Week.")
        fig_heatmap = plot_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 3. Promotion Logic (Skipped as per data inspection)
        # if 'onpromotion' in df.columns or 'discount' in df.columns:
        #     st.subheader("Promotion Impact")
        #     ...

if __name__ == "__main__":
    main()
