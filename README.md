# 🚀 Executive Sales Dashboard

A full-scale analytics platform designed to transform retail transaction data into strategic business intelligence. This dashboard provides a 360-degree view of sales performance, customer behavior, and global reach.

## 🌟 Key Features

### 1. 📈 Strategic Sales Forecast
- **Prophet Engine**: Leverages Facebook Prophet for robust time-series forecasting.
- **Dynamic Filtering**: Forecast revenue for specific **Product Lines** (e.g., Motorcycles vs. Classic Cars).
- **Interactive Horizon**: Visualize trends up to 1 year into the future.

### 2. 👥 Customer Segmentation (RFM)
- **RFM Analysis**: Automatically segments customers based on **Recency**, **Frequency**, and **Monetary** value.
- **3D Visualization**: Interactive 3D scatter plot to explore customer clusters.
- **High-Value Targets**: Instantly identifies the "Top 10" most valuable customers for VIP targeting.

### 3. 📦 Product Intelligence
- **Sunburst Chart**: Hierarchical view of sales performance from **Product Line** down to individual **Product Codes**.
- **Pricing Strategy**: Comparative analysis of **Average MSRP** vs. **Actual Selling Price** to monitor discounting behavior.

### 4. 🌍 Global Footprint
- **Geospatial Analysis**: Interactive Choropleth map showing revenue distribution across countries.
- **Territory Performance**: Breakdown of sales by global territories (EMEA, APAC, NA, etc.).

## 🛠️ Tech Stack
- **App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Prophet](https://facebook.github.io/prophet/) (Forecasting)
- **Visualization**: [Plotly](https://plotly.com/python/) (3D Scatter, Sunburst, Maps)
- **Data Processing**: [Pandas](https://pandas.pydata.org/)

## 📦 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sales-fc
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: The core application containing the Executive Dashboard logic.
- `train.csv`: Historical retail transaction data.
- `requirements.txt`: List of Python dependencies.