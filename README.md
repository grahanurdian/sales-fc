# Retail Sales Intelligence Dashboard

A production-grade analytics dashboard designed to transform retail transaction data into actionable business insights. This application combines machine learning for customer segmentation and forecasting with deep-dive analytics for product performance.

## 🚀 Key Features

### 1. 🏢 Business Insights & Segmentation
- **K-Means Clustering**: Automatically segments customers into "High Performers", "Steady Average", and "Low Volume" groups based on purchasing behavior (RFM-style analysis).
- **Visualizations**: Interactive bar charts and scatter plots to analyze sales distribution across segments.

### 2. 📈 Advanced Forecasting Engine
- **Facebook Prophet**: Powered by robust time-series modeling to handle seasonality and trends.
- **Dynamic Filtering**: Forecast sales specifically for a Customer Segment or Product Family.
- **Interactive Horizon**: Adjust the forecast period (30-365 days) with a simple slider.

### 3. 🔍 Deep Dive Analytics
- **Pareto Analysis**: Identifies the "vital few" product lines driving 80% of revenue (80/20 rule).
- **Time-Series Heatmap**: Visualizes sales intensity by Month and Day of Week to spot temporal patterns.

## 🛠️ Tech Stack
- **App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (K-Means), [Prophet](https://facebook.github.io/prophet/) (Forecasting)
- **Visualization**: [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/)

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
- `app.py`: Main application logic containing UI, data processing, and ML models.
- `train.csv`: Historical retail transaction data.
- `requirements.txt`: Python dependencies.