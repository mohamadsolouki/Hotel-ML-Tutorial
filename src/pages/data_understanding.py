"""
Data Understanding page for the Hotel Booking Analytics application.
Provides dataset overview, structure exploration, and initial insights.
"""

import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.components.ui_components import (
    render_section_header, render_info_box, render_learning_objectives,
    render_methodology_explanation, render_kpi_row, render_dataframe_preview,
    create_download_button
)
from src.utils.data_utils import get_data_summary, get_numerical_stats, get_categorical_stats
from src.utils.viz_utils import (
    plot_distribution, plot_categorical_distribution,
    plot_pie_chart, plot_correlation_matrix
)
from src.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, BUSINESS_QUESTIONS


def render_data_overview(df: pd.DataFrame):
    """Render the data overview section."""
    
    # Learning objectives
    render_learning_objectives("overview")
    
    st.markdown("---")
    
    # Dataset Summary
    render_section_header("Dataset Overview", "summarize")
    
    summary = get_data_summary(df)
    
    # Key metrics row
    metrics = [
        {'label': 'Total Records', 'value': f"{summary['total_records']:,}"},
        {'label': 'Features', 'value': str(summary['total_features'])},
        {'label': 'Countries', 'value': str(summary['countries'])},
        {'label': 'Cancellation Rate', 'value': f"{summary['cancellation_rate']:.1f}%"},
    ]
    render_kpi_row(metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # More detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Lead Time", f"{summary['avg_lead_time']:.0f} days")
        st.metric("Average Daily Rate", f"${summary['avg_adr']:.2f}")
    
    with col2:
        st.metric("Average Stay Duration", f"{summary['avg_stay_duration']:.1f} nights")
        st.metric("Date Range Start", str(summary['date_range']['start'].date()))
    
    with col3:
        st.metric("Total Est. Revenue", f"${summary['total_revenue_estimate']:,.0f}")
        st.metric("Date Range End", str(summary['date_range']['end'].date()))
    
    # Explanation
    render_info_box(
        """This dataset contains hotel booking records from two types of hotels 
        (Resort Hotel and City Hotel) covering the period from July 2015 to August 2017. 
        Each record represents a booking with 32 attributes describing the booking characteristics, 
        guest information, and reservation outcomes.""",
        title="About the Dataset"
    )


def render_data_dictionary(df: pd.DataFrame):
    """Render the data dictionary section."""
    
    render_section_header("Data Dictionary", "menu_book")
    
    render_info_box(
        """Understanding each column is crucial for effective feature engineering and model 
        development. The data dictionary below describes each attribute, its data type, and 
        its meaning in the context of hotel bookings."""
    )
    
    # Feature descriptions
    feature_info = {
        'hotel': ('Categorical', 'Type of hotel (Resort Hotel or City Hotel)'),
        'is_canceled': ('Binary', 'Whether the booking was canceled (1) or not (0) - TARGET VARIABLE'),
        'lead_time': ('Numeric', 'Number of days between booking date and arrival date'),
        'arrival_date_year': ('Numeric', 'Year of arrival date'),
        'arrival_date_month': ('Categorical', 'Month of arrival date'),
        'arrival_date_week_number': ('Numeric', 'Week number of the year for arrival date'),
        'arrival_date_day_of_month': ('Numeric', 'Day of the month of arrival date'),
        'stays_in_weekend_nights': ('Numeric', 'Number of weekend nights (Saturday/Sunday) stayed'),
        'stays_in_week_nights': ('Numeric', 'Number of week nights (Monday-Friday) stayed'),
        'adults': ('Numeric', 'Number of adults'),
        'children': ('Numeric', 'Number of children'),
        'babies': ('Numeric', 'Number of babies'),
        'meal': ('Categorical', 'Type of meal plan booked (BB=Bed & Breakfast, HB=Half Board, FB=Full Board, SC=Self Catering)'),
        'country': ('Categorical', 'Country of origin (ISO 3166-1 alpha-3 code)'),
        'market_segment': ('Categorical', 'Market segment designation (Online TA, Offline TA/TO, Direct, Corporate, etc.)'),
        'distribution_channel': ('Categorical', 'Booking distribution channel'),
        'is_repeated_guest': ('Binary', 'Whether the guest is a repeat customer (1) or not (0)'),
        'previous_cancellations': ('Numeric', 'Number of previous booking cancellations by the customer'),
        'previous_bookings_not_canceled': ('Numeric', 'Number of previous bookings not canceled'),
        'reserved_room_type': ('Categorical', 'Code of room type reserved'),
        'assigned_room_type': ('Categorical', 'Code of room type assigned'),
        'booking_changes': ('Numeric', 'Number of changes made to the booking'),
        'deposit_type': ('Categorical', 'Type of deposit made (No Deposit, Non Refund, Refundable)'),
        'agent': ('Numeric', 'ID of the travel agency that made the booking'),
        'company': ('Numeric', 'ID of the company that made the booking'),
        'days_in_waiting_list': ('Numeric', 'Number of days the booking was in waiting list'),
        'customer_type': ('Categorical', 'Type of customer (Transient, Contract, Group, Transient-Party)'),
        'adr': ('Numeric', 'Average Daily Rate - average rental income per paid occupied room'),
        'required_car_parking_spaces': ('Numeric', 'Number of car parking spaces required'),
        'total_of_special_requests': ('Numeric', 'Number of special requests made'),
        'reservation_status': ('Categorical', 'Status of the reservation (Check-Out, Canceled, No-Show)'),
        'reservation_status_date': ('Date', 'Date when the last status was set'),
    }
    
    # Create dataframe for display
    dict_df = pd.DataFrame([
        {'Column': col, 'Type': info[0], 'Description': info[1]}
        for col, info in feature_info.items()
    ])
    
    # Display with search
    search = st.text_input("Search columns:", placeholder="Type to filter...")
    if search:
        dict_df = dict_df[
            dict_df['Column'].str.contains(search, case=False) |
            dict_df['Description'].str.contains(search, case=False)
        ]
    
    st.dataframe(dict_df, use_container_width=True, height=400)
    
    # Column type summary
    st.markdown("#### Column Types Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        type_counts = dict_df['Type'].value_counts()
        st.dataframe(type_counts.reset_index().rename(
            columns={'index': 'Type', 'Type': 'Count'}
        ))
    
    with col2:
        render_info_box(
            f"""The dataset contains:
            - {len([c for c in df.columns if df[c].dtype == 'object'])} categorical columns
            - {len([c for c in df.columns if df[c].dtype in ['int64', 'float64']])} numerical columns
            - 1 date column (reservation_status_date)""",
            title="Quick Summary"
        )


def render_data_quality(df: pd.DataFrame):
    """Render the data quality assessment section."""
    
    render_section_header("Data Quality Assessment", "verified")
    
    render_methodology_explanation(
        "Data Quality Analysis",
        """Assessing data quality is essential before any analysis or modeling. 
        Poor data quality can lead to incorrect insights and unreliable models.""",
        [
            "Check for missing values in each column",
            "Identify data type inconsistencies",
            "Detect potential outliers",
            "Assess data completeness and validity"
        ]
    )
    
    # Missing values analysis
    st.markdown("#### Missing Values Analysis")
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum() / len(df) * 100).values,
        'Data Type': df.dtypes.astype(str).values
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing %', ascending=False
    )
    
    if len(missing_df) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            render_info_box(
                f"""Found {len(missing_df)} columns with missing values. 
                Total missing cells: {df.isnull().sum().sum():,} 
                ({df.isnull().sum().sum() / df.size * 100:.2f}% of all data).""",
                title="Missing Data Summary",
                box_type="warning"
            )
            
            render_info_box(
                """The 'company' column has the highest missing rate, which is expected 
                as most bookings are made by individuals rather than companies. 
                These missing values will be handled during preprocessing.""",
                title="Handling Strategy"
            )
    else:
        render_info_box(
            "No missing values detected in the dataset!",
            title="Data Complete",
            box_type="success"
        )
    
    # Data types check
    st.markdown("#### Data Types Distribution")
    
    dtype_counts = df.dtypes.value_counts()
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_pie_chart(
            pd.DataFrame({'dtype': dtype_counts.index.astype(str)}),
            'dtype',
            title="Data Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            dtype_counts.reset_index().rename(
                columns={'index': 'Data Type', 0: 'Count'}
            ),
            use_container_width=True
        )
        
        render_info_box(
            """The dataset has a good mix of numerical and categorical features. 
            Some columns like 'agent' and 'company' are stored as floats due to 
            missing values, but they represent categorical identifiers."""
        )


def render_statistical_summary(df: pd.DataFrame):
    """Render the statistical summary section."""
    
    render_section_header("Statistical Summary", "query_stats")
    
    render_methodology_explanation(
        "Descriptive Statistics",
        """Descriptive statistics provide a summary of the central tendency, 
        dispersion, and shape of the dataset's distribution.""",
        [
            "Calculate central tendency measures (mean, median, mode)",
            "Measure variability (standard deviation, variance, range)",
            "Assess distribution shape (skewness, kurtosis)",
            "Identify quartiles and potential outliers"
        ],
        formula="IQR = Q3 - Q1; Outliers: x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR"
    )
    
    tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])
    
    with tab1:
        # Select numerical column to analyze
        num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        selected_col = st.selectbox("Select a numerical column:", num_cols)
        
        if selected_col:
            stats = get_numerical_stats(df, selected_col)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig = plot_distribution(df, selected_col, bins=50)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Statistics")
                stats_df = pd.DataFrame([
                    {'Statistic': k.title(), 'Value': f"{v:,.2f}" if isinstance(v, float) else str(v)}
                    for k, v in stats.items()
                ])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Interpretation
                if abs(stats['skewness']) > 1:
                    skew_text = "highly skewed" if stats['skewness'] > 0 else "highly left-skewed"
                elif abs(stats['skewness']) > 0.5:
                    skew_text = "moderately skewed"
                else:
                    skew_text = "approximately symmetric"
                
                render_info_box(
                    f"""The distribution of {selected_col} is {skew_text}. 
                    The mean ({stats['mean']:.2f}) and median ({stats['50%']:.2f}) 
                    {"differ significantly" if abs(stats['mean'] - stats['50%']) > stats['std']*0.5 else "are close"}, 
                    indicating {"the presence of outliers" if abs(stats['mean'] - stats['50%']) > stats['std']*0.5 else "a balanced distribution"}."""
                )
    
    with tab2:
        # Select categorical column to analyze
        cat_cols = CATEGORICAL_COLUMNS + ['stay_type', 'lead_time_category']
        cat_cols = [col for col in cat_cols if col in df.columns]
        selected_cat = st.selectbox("Select a categorical column:", cat_cols)
        
        if selected_cat:
            stats = get_categorical_stats(df, selected_cat)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig = plot_categorical_distribution(
                    df, selected_cat, 
                    horizontal=True if stats['unique'] > 5 else False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Statistics")
                st.metric("Unique Values", stats['unique'])
                st.metric("Most Common", stats['top'])
                st.metric("Frequency", f"{stats['top_freq']:,} ({stats['top_pct']:.1f}%)")
                
                if stats['missing'] > 0:
                    st.metric("Missing", f"{stats['missing']:,} ({stats['missing_pct']:.1f}%)")


def render_business_questions(df: pd.DataFrame):
    """Render the business questions section."""
    
    render_section_header("Business Questions", "help_center")
    
    render_info_box(
        """Before diving into analysis, it's crucial to identify the key business questions 
        we want to answer. This dataset allows us to explore various aspects of hotel booking 
        behavior, seasonality, cancellations, and revenue patterns.""",
        title="Defining Our Goals"
    )
    
    for category, questions in BUSINESS_QUESTIONS.items():
        with st.expander(f"{category}", expanded=False):
            for i, question in enumerate(questions, 1):
                st.markdown(f"**Q{i}.** {question}")
            
            # Add relevance to dataset
            st.markdown("---")
            st.markdown("**Relevant Features:**")
            
            if "Seasonal" in category:
                st.markdown("`arrival_date_year`, `arrival_date_month`, `arrival_date_week_number`")
            elif "Demand" in category:
                st.markdown("`lead_time`, `is_canceled`, `arrival_date_*`, `market_segment`")
            elif "Cancellation" in category:
                st.markdown("`is_canceled`, `lead_time`, `deposit_type`, `previous_cancellations`")
            elif "Revenue" in category:
                st.markdown("`adr`, `stays_in_*`, `market_segment`, `customer_type`")
            elif "Customer" in category:
                st.markdown("`is_repeated_guest`, `customer_type`, `country`, `total_of_special_requests`")


def render_data_preview(df: pd.DataFrame):
    """Render the data preview section."""
    
    render_section_header("Data Preview", "preview")
    
    # Row count selector
    n_rows = st.slider("Number of rows to display:", 5, 100, 10)
    
    # Column filter
    all_cols = df.columns.tolist()
    selected_cols = st.multiselect(
        "Select columns to display:",
        all_cols,
        default=all_cols[:10]
    )
    
    if selected_cols:
        st.dataframe(df[selected_cols].head(n_rows), use_container_width=True)
    
    # Download option
    col1, col2 = st.columns([1, 4])
    with col1:
        create_download_button(df.head(1000), "hotel_bookings_sample.csv", "Download Sample (1000 rows)")


def render_page(df: pd.DataFrame):
    """Main render function for the Data Understanding page."""
    
    tabs = st.tabs([
        "Overview",
        "Data Dictionary", 
        "Data Quality",
        "Statistics",
        "Business Questions",
        "Data Preview"
    ])
    
    with tabs[0]:
        render_data_overview(df)
    
    with tabs[1]:
        render_data_dictionary(df)
    
    with tabs[2]:
        render_data_quality(df)
    
    with tabs[3]:
        render_statistical_summary(df)
    
    with tabs[4]:
        render_business_questions(df)
    
    with tabs[5]:
        render_data_preview(df)
