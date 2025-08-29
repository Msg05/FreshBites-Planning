# freshbites_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import plotly.express as px
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="FreshBites Supply Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px;}
    .risk-critical {color: #ff4b4b; font-weight: bold;}
    .risk-warning {color: #ffa500; font-weight: bold;}
    .risk-good {color: #00cc96; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📦 FreshBites Supply Chain Optimizer</h1>', unsafe_allow_html=True)
st.write("AI-powered supply chain optimization for demand forecasting and production planning")

# Sample data generation function
@st.cache_data
def load_data():
    np.random.seed(42)
    weeks = 35
    skus = ['Potato Chips', 'Nachos', 'Cookies', 'Energy Bar', 'Instant Noodles']
    regions = ['Mumbai', 'Kolkata', 'Delhi']
    
    dates = pd.date_range(start='2023-01-01', periods=weeks, freq='W-SUN')
    data = []
    
    for week_id, date in enumerate(dates, 1):
        is_festival = 1 if week_id in [5, 10, 15, 20, 25, 30, 35] else 0
        
        for sku in skus:
            for region in regions:
                base_demand = np.random.randint(20, 40)
                
                if region == 'Mumbai':
                    base_demand *= 1.4
                elif region == 'Kolkata' and sku != 'Cookies':
                    base_demand *= 0.7

                forecast = base_demand + np.random.randint(-5, 5)
                actual = base_demand * (1 + is_festival * np.random.uniform(0.4, 0.5))
                actual += np.random.randint(-7, 7)
                actual = max(0, actual)
                
                if region == 'Mumbai':
                    current_stock = max(5, np.random.randint(0, 15))
                elif region == 'Kolkata' and sku == 'Cookies':
                    current_stock = np.random.randint(30, 50)
                else:
                    current_stock = np.random.randint(10, 25)
                
                data.append({
                    'Week_ID': week_id,
                    'Date': date,
                    'SKU': sku,
                    'Region': region,
                    'Forecast_Demand': round(forecast, 2),
                    'Actual_Demand': round(actual, 2),
                    'Current_Stock': current_stock,
                    'Is_Festival': is_festival,
                    'Plant': 'Delhi' if region in ['Delhi', 'Kolkata'] else 'Pune'
                })
    
    df = pd.DataFrame(data)
    
    # Create adjusted forecast
    def create_adjusted_forecast(row):
        base_forecast = row['Forecast_Demand']
        if row['Is_Festival'] == 1:
            base_forecast *= 1.45
        if row['Region'] == 'Mumbai':
            base_forecast *= 1.10
        elif row['Region'] == 'Kolkata' and row['SKU'] != 'Cookies':
            base_forecast *= 0.80
        return max(5, base_forecast)
    
    df['Adjusted_Forecast'] = df.apply(create_adjusted_forecast, axis=1)
    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("🔧 Control Panel")
selected_week = st.sidebar.selectbox("Select Week", sorted(df['Week_ID'].unique()))
selected_region = st.sidebar.multiselect("Select Regions", df['Region'].unique(), default=df['Region'].unique())
selected_sku = st.sidebar.multiselect("Select SKUs", df['SKU'].unique(), default=df['SKU'].unique())

# Filter data
filtered_df = df[
    (df['Week_ID'] == selected_week) & 
    (df['Region'].isin(selected_region)) & 
    (df['SKU'].isin(selected_sku))
]

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Demand", f"{filtered_df['Adjusted_Forecast'].sum():.0f} units")
with col2:
    st.metric("Current Stock", f"{filtered_df['Current_Stock'].sum():.0f} units")
with col3:
    stock_out_risk = (filtered_df['Current_Stock'] < filtered_df['Adjusted_Forecast'] * 0.5).sum()
    st.metric("Stock-out Risks", f"{stock_out_risk}")
with col4:
    festival_status = "Yes" if filtered_df['Is_Festival'].iloc[0] == 1 else "No"
    st.metric("Festival Week", festival_status)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Optimization", "📈 Analytics"])

with tab1:
    st.subheader("Demand vs Stock Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use Streamlit's native bar chart
        chart_data = filtered_df.groupby('SKU')[['Adjusted_Forecast', 'Current_Stock']].sum().reset_index()
        st.bar_chart(chart_data.set_index('SKU'))
    
    with col2:
        region_summary = filtered_df.groupby('Region')[['Adjusted_Forecast', 'Current_Stock']].sum().reset_index()
        st.bar_chart(region_summary.set_index('Region'))

with tab2:
    st.subheader("Production Optimization")
    
    # Simplified optimization function
    def run_optimization(week_data):
        skus = week_data['SKU'].unique()
        plants = week_data['Plant'].unique()
        demand_dict = week_data.groupby('SKU')['Adjusted_Forecast'].first().to_dict()
        stock_dict = week_data.groupby('SKU')['Current_Stock'].first().to_dict()
        capacity_dict = {'Delhi': 100, 'Pune': 80}
        
        prob = LpProblem("Production_Optimization", LpMinimize)
        production_vars = LpVariable.dicts("Prod", ((s, p) for s in skus for p in plants), 0)
        stock_out_vars = LpVariable.dicts("StockOut", skus, 0)
        
        prob += lpSum(stock_out_vars[s] for s in skus)
        
        for plant in plants:
            prob += lpSum(production_vars[s, plant] for s in skus) <= capacity_dict[plant]
        
        for sku in skus:
            total_prod = lpSum(production_vars[sku, p] for p in plants)
            prob += total_prod + stock_dict[sku] + stock_out_vars[sku] >= demand_dict[sku]
        
        prob.solve(PULP_CBC_CMD(msg=False))
        
        if LpStatus[prob.status] == 'Optimal':
            results = []
            for sku in skus:
                for plant in plants:
                    if production_vars[sku, plant].varValue > 0:
                        results.append({
                            'SKU': sku,
                            'Plant': plant,
                            'Production': production_vars[sku, plant].varValue,
                            'Demand': demand_dict[sku],
                            'Current_Stock': stock_dict[sku]
                        })
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Optimizing production plan..."):
            results = run_optimization(filtered_df)
            
            if not results.empty:
                st.success("Optimization complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(results.style.format({
                        'Production': '{:.1f}',
                        'Demand': '{:.1f}',
                        'Current_Stock': '{:.0f}'
                    }))
                
                with col2:
                    # Use Streamlit native bar chart
                    production_summary = results.groupby('SKU')['Production'].sum().reset_index()
                    st.bar_chart(production_summary.set_index('SKU'))
            else:
                st.warning("No optimization needed or solution not found")

with tab3:
    st.subheader("Analytics & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast accuracy over time
        accuracy_df = df.groupby('Week_ID').agg({
            'Forecast_Demand': 'sum',
            'Actual_Demand': 'sum',
            'Adjusted_Forecast': 'sum'
        }).reset_index()
        
        st.line_chart(accuracy_df.set_index('Week_ID'))
    
    with col2:
        # Inventory health
        inventory_health = filtered_df.copy()
        inventory_health['Status'] = np.where(
            inventory_health['Current_Stock'] < inventory_health['Adjusted_Forecast'] * 0.5,
            'Stock-out Risk',
            np.where(
                inventory_health['Current_Stock'] > inventory_health['Adjusted_Forecast'] * 1.5,
                'Overstock',
                'Optimal'
            )
        )
        
        status_count = inventory_health['Status'].value_counts()
        st.write("**Inventory Health Status**")
        for status, count in status_count.items():
            st.write(f"- {status}: {count} items")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**FreshBites Supply Optimizer**  
AI-powered supply chain management  
🚀 Built for Hackathon 2024
""")
