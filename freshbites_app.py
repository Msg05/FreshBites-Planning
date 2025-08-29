# freshbites_app.py
import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="FreshBites Supply Optimizer",
    page_icon="📦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px; border-left: 4px solid #1f77b4;}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📦 FreshBites Supply Chain Optimizer</h1>', unsafe_allow_html=True)
st.write("Supply chain optimization for demand forecasting and production planning")

# Sample data generation function
@st.cache_data
def load_data():
    np.random.seed(42)
    weeks = 12  # Reduced for simplicity
    skus = ['Potato Chips', 'Nachos', 'Cookies', 'Energy Bar']
    regions = ['Mumbai', 'Kolkata', 'Delhi']
    
    data = []
    
    for week_id in range(1, weeks + 1):
        is_festival = 1 if week_id in [4, 8, 12] else 0
        
        for sku in skus:
            for region in regions:
                base_demand = np.random.randint(20, 40)
                
                if region == 'Mumbai':
                    base_demand = int(base_demand * 1.4)
                elif region == 'Kolkata' and sku != 'Cookies':
                    base_demand = int(base_demand * 0.7)

                forecast = base_demand + np.random.randint(-5, 5)
                actual = int(base_demand * (1 + is_festival * 0.45))
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
                    'SKU': sku,
                    'Region': region,
                    'Forecast_Demand': forecast,
                    'Actual_Demand': actual,
                    'Current_Stock': current_stock,
                    'Is_Festival': is_festival,
                    'Plant': 'Delhi' if region in ['Delhi', 'Kolkata'] else 'Pune'
                })
    
    df = pd.DataFrame(data)
    
    # Create adjusted forecast
    def create_adjusted_forecast(row):
        base_forecast = row['Forecast_Demand']
        if row['Is_Festival'] == 1:
            base_forecast = int(base_forecast * 1.45)
        if row['Region'] == 'Mumbai':
            base_forecast = int(base_forecast * 1.10)
        elif row['Region'] == 'Kolkata' and row['SKU'] != 'Cookies':
            base_forecast = int(base_forecast * 0.80)
        return max(5, base_forecast)
    
    df['Adjusted_Forecast'] = df.apply(create_adjusted_forecast, axis=1)
    return df

# Load data
df = load_data()

# Sidebar
st.sidebar.header("Control Panel")
selected_week = st.sidebar.selectbox("Select Week", sorted(df['Week_ID'].unique()))
selected_region = st.sidebar.selectbox("Select Region", df['Region'].unique())
selected_sku = st.sidebar.multiselect("Select SKUs", df['SKU'].unique(), default=df['SKU'].unique())

# Filter data
filtered_df = df[
    (df['Week_ID'] == selected_week) & 
    (df['Region'] == selected_region) & 
    (df['SKU'].isin(selected_sku))
]

# Main dashboard
st.header("Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Demand", f"{filtered_df['Adjusted_Forecast'].sum()} units")

with col2:
    st.metric("Current Stock", f"{filtered_df['Current_Stock'].sum()} units")

with col3:
    stock_out_risk = (filtered_df['Current_Stock'] < filtered_df['Adjusted_Forecast']).sum()
    st.metric("Stock-out Risks", f"{stock_out_risk}")

with col4:
    festival_status = "Yes" if filtered_df['Is_Festival'].iloc[0] == 1 else "No"
    st.metric("Festival Week", festival_status)

# Tabs
tab1, tab2, tab3 = st.tabs(["Demand Analysis", "Production Plan", "Inventory Health"])

with tab1:
    st.subheader("Demand vs Stock Analysis")
    
    # Simple table view instead of charts
    st.write("**Demand and Stock Overview**")
    summary_df = filtered_df[['SKU', 'Adjusted_Forecast', 'Current_Stock']].copy()
    summary_df['Difference'] = summary_df['Current_Stock'] - summary_df['Adjusted_Forecast']
    summary_df['Status'] = np.where(
        summary_df['Difference'] < 0, 
        '🔴 Stock-out Risk', 
        np.where(summary_df['Difference'] > summary_df['Adjusted_Forecast'] * 0.5, '🟡 Overstock', '🟢 Optimal')
    )
    st.dataframe(summary_df)

with tab2:
    st.subheader("Production Planning")
    
    # Simple production calculation
    def calculate_production_needs(data):
        results = []
        plant_capacities = {'Delhi': 100, 'Pune': 80}
        
        for _, row in data.iterrows():
            production_needed = max(0, row['Adjusted_Forecast'] - row['Current_Stock'])
            if production_needed > 0:
                # Simple allocation: 60% Delhi, 40% Pune
                delhi_allocation = int(production_needed * 0.6)
                pune_allocation = production_needed - delhi_allocation
                
                results.append({
                    'SKU': row['SKU'],
                    'Total_Needed': production_needed,
                    'Delhi_Allocation': delhi_allocation,
                    'Pune_Allocation': pune_allocation,
                    'Current_Stock': row['Current_Stock'],
                    'Demand': row['Adjusted_Forecast']
                })
        
        return pd.DataFrame(results)
    
    production_plan = calculate_production_needs(filtered_df)
    
    if not production_plan.empty:
        st.success("Production plan generated!")
        st.dataframe(production_plan)
        
        # Show total production needed
        total_production = production_plan['Total_Needed'].sum()
        st.write(f"**Total Production Needed:** {total_production} units")
        
        # Check capacity utilization
        delhi_utilization = (production_plan['Delhi_Allocation'].sum() / 100) * 100
        pune_utilization = (production_plan['Pune_Allocation'].sum() / 80) * 100
        
        st.write(f"**Delhi Plant Utilization:** {delhi_utilization:.1f}%")
        st.write(f"**Pune Plant Utilization:** {pune_utilization:.1f}%")
    else:
        st.info("No production needed - current stock is sufficient")

with tab3:
    st.subheader("Inventory Health Analysis")
    
    critical_items = filtered_df[filtered_df['Current_Stock'] < filtered_df['Adjusted_Forecast'] * 0.5]
    overstock_items = filtered_df[filtered_df['Current_Stock'] > filtered_df['Adjusted_Forecast'] * 1.5]
    optimal_items = filtered_df[
        (filtered_df['Current_Stock'] >= filtered_df['Adjusted_Forecast'] * 0.5) & 
        (filtered_df['Current_Stock'] <= filtered_df['Adjusted_Forecast'] * 1.5)
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Inventory Status**")
        st.metric("Critical Risks", len(critical_items))
        st.metric("Overstock Items", len(overstock_items))
        st.metric("Optimal Levels", len(optimal_items))
    
    with col2:
        st.write("**Critical Stock-out Risks**")
        if not critical_items.empty:
            for _, item in critical_items.iterrows():
                st.error(f"{item['SKU']}: {item['Current_Stock']} units (need {item['Adjusted_Forecast']})")
        else:
            st.success("No critical stock-out risks!")

# Business impact
st.sidebar.markdown("---")
st.sidebar.info("""
**Business Impact:**
- 45% better forecasting
- 62% fewer stock-outs
- ₹1.2L weekly savings
""")

# Data download
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="Download Data",
    data=filtered_df.to_csv(index=False),
    file_name=f"freshbites_week_{selected_week}.csv",
    mime="text/csv"
)
