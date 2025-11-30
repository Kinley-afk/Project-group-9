import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Bhutan TB Analytics", layout="wide")

st.title("🇧🇹 Bhutan Tuberculosis Healthcare Analytics")
st.write("Comprehensive analysis of tuberculosis indicators in Bhutan using WHO data")

# ----------------------------------------------------
# SECTION 1: DATA LOADING AND BASIC INFO
# ----------------------------------------------------
st.header("1. Tuberculosis Dataset Overview")

# Load the data
df = pd.read_csv("tuberculosis_indicators_btn.csv")

st.write("### Dataset Preview")
st.dataframe(df.head(10))

st.write("### Dataset Shape")
st.write(f"Number of records: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")

# ----------------------------------------------------
# SECTION 2: DATA CLEANING AND PREPROCESSING
# ----------------------------------------------------
st.header("2. Data Cleaning & Preprocessing")

# Create a copy for cleaning
df_clean = df.copy()

# Basic cleaning
st.write("### Cleaning Steps Applied:")
cleaning_steps = []

# Handle missing values in numeric columns
numeric_cols = ['Numeric', 'Low', 'High']
for col in numeric_cols:
    if col in df_clean.columns:
        initial_missing = df_clean[col].isna().sum()
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
        cleaning_steps.append(f"Filled {initial_missing} missing values in {col}")

# Clean year column
df_clean['YEAR (DISPLAY)'] = pd.to_numeric(df_clean['YEAR (DISPLAY)'], errors='coerce')
df_clean = df_clean.dropna(subset=['YEAR (DISPLAY)'])
df_clean['YEAR (DISPLAY)'] = df_clean['YEAR (DISPLAY)'].astype(int)

# Remove duplicates
initial_rows = len(df)
df_clean = df_clean.drop_duplicates()
duplicates_removed = initial_rows - len(df_clean)
if duplicates_removed > 0:
    cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows")

# Display cleaning steps
for step in cleaning_steps:
    st.write(f"✅ {step}")

st.write("### Cleaned Data Preview")
st.dataframe(df_clean.head())

# ----------------------------------------------------
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------
st.header("3. Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("#### TB Indicators Summary")
    indicator_counts = df_clean['GHO (DISPLAY)'].value_counts()
    st.write(f"Unique TB indicators: {len(indicator_counts)}")
    st.dataframe(indicator_counts.head(10))

with col2:
    st.write("#### Year Range")
    min_year = df_clean['YEAR (DISPLAY)'].min()
    max_year = df_clean['YEAR (DISPLAY)'].max()
    st.write(f"Data spans from {min_year} to {max_year}")

# Time trend analysis
st.write("#### TB Incidence Trends Over Time")

# Get main incidence indicators
incidence_indicators = [
    'Incidence of tuberculosis (per 100 000 population per year)',
    'Number of incident tuberculosis cases',
    'Number of incident tuberculosis cases,  (HIV-positive cases)'
]

selected_indicator = st.selectbox("Select indicator to visualize:", incidence_indicators)

if selected_indicator:
    indicator_data = df_clean[df_clean['GHO (DISPLAY)'] == selected_indicator]
    
    if not indicator_data.empty:
        # Sort by year
        indicator_data = indicator_data.sort_values('YEAR (DISPLAY)')
        
        # Create chart data
        chart_data = indicator_data[['YEAR (DISPLAY)', 'Numeric']].set_index('YEAR (DISPLAY)')
        
        st.write(f"**Trend of {selected_indicator}**")
        st.line_chart(chart_data)
        
        # Show data table
        st.write("**Data Values:**")
        st.dataframe(indicator_data[['YEAR (DISPLAY)', 'Numeric', 'Low', 'High']].reset_index(drop=True))

# Treatment success rates
st.write("#### Treatment Success Rates")
treatment_indicators = [ind for ind in df_clean['GHO (DISPLAY)'].unique() 
                      if 'Treatment success rate' in ind]

if treatment_indicators:
    selected_treatment = st.selectbox("Select treatment indicator:", treatment_indicators)
    
    if selected_treatment:
        treatment_data = df_clean[df_clean['GHO (DISPLAY)'] == selected_treatment]
        treatment_data = treatment_data.sort_values('YEAR (DISPLAY)')
        
        if not treatment_data.empty:
            chart_data = treatment_data[['YEAR (DISPLAY)', 'Numeric']].set_index('YEAR (DISPLAY)')
            st.write(f"**{selected_treatment} Over Time**")
            st.line_chart(chart_data)

# ----------------------------------------------------
# SECTION 4: FEATURE ENGINEERING
# ----------------------------------------------------
st.header("4. Feature Engineering")

df_features = df_clean.copy()

# Create aggregated features by year
yearly_aggregates = df_features.groupby('YEAR (DISPLAY)').agg({
    'Numeric': ['mean', 'std', 'count']
}).round(2)
yearly_aggregates.columns = ['Average_Value', 'Std_Dev', 'Indicator_Count']

st.write("#### Yearly Aggregated Statistics")
st.dataframe(yearly_aggregates)

# Create indicator categories
def categorize_indicator(indicator_name):
    indicator_lower = indicator_name.lower()
    if 'incidence' in indicator_lower:
        return 'Incidence'
    elif 'treatment' in indicator_lower or 'success' in indicator_lower:
        return 'Treatment'
    elif 'test' in indicator_lower or 'hiv' in indicator_lower:
        return 'Testing'
    elif 'mdr' in indicator_lower or 'rr' in indicator_lower:
        return 'Drug_Resistance'
    elif 'case' in indicator_lower:
        return 'Cases'
    else:
        return 'Other'

df_features['Indicator_Category'] = df_features['GHO (DISPLAY)'].apply(categorize_indicator)

st.write("#### Indicator Categories")
category_counts = df_features['Indicator_Category'].value_counts()
st.dataframe(category_counts)

# Show category distribution using bar chart
st.write("#### Category Distribution")
category_chart_data = pd.DataFrame({
    'Category': category_counts.index,
    'Count': category_counts.values
})
st.bar_chart(category_chart_data.set_index('Category'))

# ----------------------------------------------------
# SECTION 5: MACHINE LEARNING MODEL
# ----------------------------------------------------
st.header("5. TB Prediction Model")

# Prepare data for modeling
model_data = df_features[df_features['GHO (DISPLAY)'] == 'Incidence of tuberculosis (per 100 000 population per year)']
model_data = model_data[['YEAR (DISPLAY)', 'Numeric', 'Low', 'High']].dropna()

if len(model_data) > 5:  # Ensure we have enough data
    # Create features for prediction
    X = model_data[['YEAR (DISPLAY)']]
    y = model_data['Numeric']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Model Performance")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        
        # Show actual vs predicted in a table
        comparison_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred,
            'Year': X_test['YEAR (DISPLAY)'].values
        })
        st.write("**Actual vs Predicted Values:**")
        st.dataframe(comparison_df)
        
        # Future prediction
        st.write("#### Predict Future Incidence")
        future_year = st.number_input("Enter year for prediction:", 
                                   min_value=2025, max_value=2035, value=2025)
        
        if st.button("Predict TB Incidence"):
            prediction = model.predict([[future_year]])[0]
            st.success(f"Predicted TB incidence in {future_year}: {prediction:.0f} per 100,000 population")
    
    with col2:
        # Show model feature importance
        st.write("#### Model Insights")
        importance_df = pd.DataFrame({
            'Feature': ['Year'],
            'Importance': [model.feature_importances_[0]]
        })
        st.write("**Feature Importance:**")
        st.dataframe(importance_df)
        
        # Show training data summary
        st.write("**Training Data Summary:**")
        st.write(f"Training samples: {len(X_train)}")
        st.write(f"Test samples: {len(X_test)}")
        st.write(f"Years in training: {X_train['YEAR (DISPLAY)'].min()} - {X_train['YEAR (DISPLAY)'].max()}")
        
else:
    st.warning("Not enough data for reliable modeling. Need more incidence data points.")

# ----------------------------------------------------
# SECTION 6: KEY INSIGHTS DASHBOARD
# ----------------------------------------------------
st.header("6. Key Insights Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    # Latest incidence rate
    latest_incidence = df_features[
        df_features['GHO (DISPLAY)'] == 'Incidence of tuberculosis (per 100 000 population per year)'
    ].sort_values('YEAR (DISPLAY)').iloc[-1] if not df_features[
        df_features['GHO (DISPLAY)'] == 'Incidence of tuberculosis (per 100 000 population per year)'
    ].empty else None
    
    if latest_incidence is not None:
        st.metric(
            label=f"TB Incidence ({latest_incidence['YEAR (DISPLAY)']})",
            value=f"{latest_incidence['Numeric']:.0f}",
            delta=f"per 100,000 population"
        )

with col2:
    # Treatment success rate
    latest_treatment = df_features[
        df_features['GHO (DISPLAY)'] == 'Treatment success rate: new TB cases'
    ].sort_values('YEAR (DISPLAY)').iloc[-1] if not df_features[
        df_features['GHO (DISPLAY)'] == 'Treatment success rate: new TB cases'
    ].empty else None
    
    if latest_treatment is not None:
        st.metric(
            label=f"Treatment Success Rate ({latest_treatment['YEAR (DISPLAY)']})",
            value=f"{latest_treatment['Numeric']}%"
        )

with col3:
    # MDR-TB cases
    latest_mdr = df_features[
        df_features['GHO (DISPLAY)'] == 'Confirmed cases of RR-/MDR-TB'
    ].sort_values('YEAR (DISPLAY)').iloc[-1] if not df_features[
        df_features['GHO (DISPLAY)'] == 'Confirmed cases of RR-/MDR-TB'
    ].empty else None
    
    if latest_mdr is not None:
        st.metric(
            label=f"MDR-TB Cases ({latest_mdr['YEAR (DISPLAY)']})",
            value=f"{latest_mdr['Numeric']:.0f}"
        )

# Additional metrics
st.write("#### Additional Key Metrics")
col4, col5, col6 = st.columns(3)

with col4:
    # HIV testing coverage
    hiv_testing = df_features[
        df_features['GHO (DISPLAY)'] == 'TB patients with known HIV status (%)'
    ].sort_values('YEAR (DISPLAY)').iloc[-1] if not df_features[
        df_features['GHO (DISPLAY)'] == 'TB patients with known HIV status (%)'
    ].empty else None
    
    if hiv_testing is not None:
        st.metric(
            label=f"HIV Testing Coverage ({hiv_testing['YEAR (DISPLAY)']})",
            value=f"{hiv_testing['Numeric']}%"
        )

with col5:
    # MDR-TB treatment success
    mdr_treatment = df_features[
        df_features['GHO (DISPLAY)'] == 'Treatment success rate for patients treated for MDR-TB (%)'
    ].sort_values('YEAR (DISPLAY)').iloc[-1] if not df_features[
        df_features['GHO (DISPLAY)'] == 'Treatment success rate for patients treated for MDR-TB (%)'
    ].empty else None
    
    if mdr_treatment is not None:
        st.metric(
            label=f"MDR-TB Treatment Success ({mdr_treatment['YEAR (DISPLAY)']})",
            value=f"{mdr_treatment['Numeric']}%"
        )

with col6:
    # Total cases trend
    current_cases = df_features[
        df_features['GHO (DISPLAY)'] == 'Number of incident tuberculosis cases'
    ].sort_values('YEAR (DISPLAY)').iloc[-1] if not df_features[
        df_features['GHO (DISPLAY)'] == 'Number of incident tuberculosis cases'
    ].empty else None
    
    previous_cases = df_features[
        df_features['GHO (DISPLAY)'] == 'Number of incident tuberculosis cases'
    ].sort_values('YEAR (DISPLAY)').iloc[-2] if len(df_features[
        df_features['GHO (DISPLAY)'] == 'Number of incident tuberculosis cases'
    ]) > 1 else None
    
    if current_cases is not None:
        delta_value = None
        if previous_cases is not None:
            delta_value = current_cases['Numeric'] - previous_cases['Numeric']
        
        st.metric(
            label=f"Total TB Cases ({current_cases['YEAR (DISPLAY)']})",
            value=f"{current_cases['Numeric']:.0f}",
            delta=delta_value
        )

# ----------------------------------------------------
# SECTION 7: DATA EXPORT
# ----------------------------------------------------
st.header("7. Export Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("Download Cleaned Dataset"):
        csv = df_clean.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="bhutan_tb_cleaned_data.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Download Yearly Summary"):
        csv = yearly_aggregates.to_csv()
        st.download_button(
            label="Download Summary CSV",
            data=csv,
            file_name="bhutan_tb_yearly_summary.csv",
            mime="text/csv"
        )

# ----------------------------------------------------
# SECTION 8: RECOMMENDATIONS
# ----------------------------------------------------
st.header("8. Public Health Insights")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.write("### 📈 Positive Trends")
    st.write("""
    - **Declining Incidence**: TB incidence rates have shown significant decrease over time
    - **High Treatment Success**: Consistently high treatment success rates for new cases
    - **Improved Testing**: Increased HIV testing among TB patients in recent years
    - **MDR-TB Management**: Better detection and treatment of drug-resistant TB
    - **Enhanced Coverage**: Improved tuberculosis treatment coverage over the years
    """)

with insights_col2:
    st.write("### 🎯 Focus Areas")
    st.write("""
    - **Early Detection**: Continue improving case detection rates
    - **MDR-TB Control**: Maintain vigilance on drug-resistant strains
    - **HIV Integration**: Strengthen TB-HIV collaborative activities
    - **Community Engagement**: Enhance community-based TB care
    - **Data Continuity**: Ensure consistent monitoring and reporting
    """)

st.write("### 📊 Data Quality Assessment")
st.write("""
- **Comprehensive Coverage**: Multiple TB indicators tracked over time
- **Recent Data**: Includes data up to 2024 for most indicators
- **Consistent Metrics**: Standardized WHO definitions and methodologies
- **Confidence Intervals**: Many estimates include uncertainty ranges
""")

st.write("---")
st.write("**Data Source**: WHO Global Health Observatory - Tuberculosis indicators for Bhutan")
st.write("**Last Updated**: Dataset includes data up to 2024")
st.write("**Analysis Period**: Comprehensive analysis from 2000 to 2024")
