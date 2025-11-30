import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Load the data with file uploader
uploaded_file = st.file_uploader("Upload tuberculosis_indicators_btn.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
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
    
    # Filter to only show indicators that exist in the data
    available_indicators = [ind for ind in incidence_indicators if ind in df_clean['GHO (DISPLAY)'].values]
    
    if available_indicators:
        selected_indicator = st.selectbox("Select indicator to visualize:", available_indicators)
        
        if selected_indicator:
            indicator_data = df_clean[df_clean['GHO (DISPLAY)'] == selected_indicator]
            
            if not indicator_data.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Sort by year
                indicator_data = indicator_data.sort_values('YEAR (DISPLAY)')
                
                # Plot main values
                ax.plot(indicator_data['YEAR (DISPLAY)'], indicator_data['Numeric'], 
                        marker='o', linewidth=2, markersize=6, label='Value')
                
                # Plot confidence intervals if available
                if 'Low' in indicator_data.columns and 'High' in indicator_data.columns:
                    ax.fill_between(indicator_data['YEAR (DISPLAY)'], 
                                   indicator_data['Low'], 
                                   indicator_data['High'], 
                                   alpha=0.3, label='Confidence Interval')
                
                ax.set_xlabel('Year')
                ax.set_ylabel(selected_indicator)
                ax.set_title(f'Trend of {selected_indicator} in Bhutan')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
    
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
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(treatment_data['YEAR (DISPLAY)'], treatment_data['Numeric'], 
                       marker='s', linewidth=2, markersize=6, color='green')
                ax.set_xlabel('Year')
                ax.set_ylabel('Success Rate (%)')
                ax.set_title(f'{selected_treatment} Over Time')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)
                
                st.pyplot(fig)
    
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
            
            # Future prediction
            st.write("#### Predict Future Incidence")
            future_year = st.number_input("Enter year for prediction:", 
                                       min_value=2025, max_value=2035, value=2025)
            
            if st.button("Predict TB Incidence"):
                prediction = model.predict([[future_year]])[0]
                st.success(f"Predicted TB incidence in {future_year}: {prediction:.0f} per 100,000 population")
        
        with col2:
            # Plot actual vs predicted
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted TB Incidence')
            st.pyplot(fig)
            
    else:
        st.warning("Not enough data for reliable modeling. Need more incidence data points.")
    
    # ----------------------------------------------------
    # SECTION 6: KEY INSIGHTS DASHBOARD
    # ----------------------------------------------------
    st.header("6. Key Insights Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Latest incidence rate
        incidence_data = df_features[
            df_features['GHO (DISPLAY)'] == 'Incidence of tuberculosis (per 100 000 population per year)'
        ]
        if not incidence_data.empty:
            latest_incidence = incidence_data.sort_values('YEAR (DISPLAY)').iloc[-1]
            st.metric(
                label=f"TB Incidence ({latest_incidence['YEAR (DISPLAY)']})",
                value=f"{latest_incidence['Numeric']:.0f}",
                delta=f"per 100,000 population"
            )
        else:
            st.metric(
                label="TB Incidence",
                value="N/A",
                delta="Data not available"
            )
    
    with col2:
        # Treatment success rate
        treatment_data = df_features[
            df_features['GHO (DISPLAY)'] == 'Treatment success rate: new TB cases'
        ]
        if not treatment_data.empty:
            latest_treatment = treatment_data.sort_values('YEAR (DISPLAY)').iloc[-1]
            st.metric(
                label=f"Treatment Success Rate ({latest_treatment['YEAR (DISPLAY)']})",
                value=f"{latest_treatment['Numeric']}%"
            )
        else:
            st.metric(
                label="Treatment Success Rate",
                value="N/A",
                delta="Data not available"
            )
    
    with col3:
        # MDR-TB cases
        mdr_data = df_features[
            df_features['GHO (DISPLAY)'] == 'Confirmed cases of RR-/MDR-TB'
        ]
        if not mdr_data.empty:
            latest_mdr = mdr_data.sort_values('YEAR (DISPLAY)').iloc[-1]
            st.metric(
                label=f"MDR-TB Cases ({latest_mdr['YEAR (DISPLAY)']})",
                value=f"{latest_mdr['Numeric']:.0f}"
            )
        else:
            st.metric(
                label="MDR-TB Cases",
                value="N/A",
                delta="Data not available"
            )
    
    # Additional visualizations
    st.write("#### Category Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    category_counts = df_features['Indicator_Category'].value_counts()
    ax.bar(category_counts.index, category_counts.values)
    ax.set_xlabel('Indicator Categories')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of TB Indicator Categories')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # ----------------------------------------------------
    # SECTION 7: DATA EXPORT
    # ----------------------------------------------------
    st.header("7. Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_clean = df_clean.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Dataset",
            data=csv_clean,
            file_name="bhutan_tb_cleaned_data.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_summary = yearly_aggregates.to_csv()
        st.download_button(
            label="Download Yearly Summary",
            data=csv_summary,
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
        """)
    
    with insights_col2:
        st.write("### 🎯 Focus Areas")
        st.write("""
        - **Early Detection**: Continue improving case detection rates
        - **MDR-TB Control**: Maintain vigilance on drug-resistant strains
        - **HIV Integration**: Strengthen TB-HIV collaborative activities
        - **Community Engagement**: Enhance community-based TB care
        """)
    
    st.write("---")
    st.write("**Data Source**: WHO Global Health Observatory - Tuberculosis indicators for Bhutan")
    st.write("**Last Updated**: Dataset includes data up to 2024")
    
else:
    st.warning("📁 Please upload the tuberculosis_indicators_btn.csv file to begin analysis.")
    st.info("""
    **How to use this app:**
    1. Click 'Browse files' above
    2. Select your tuberculosis_indicators_btn.csv file
    3. The analysis will automatically load
    
    **Expected CSV format:**
    - Should contain TB indicators from WHO data
    - Include columns like 'GHO (DISPLAY)', 'YEAR (DISPLAY)', 'Numeric', etc.
    - Data should span multiple years for trend analysis
    """)
