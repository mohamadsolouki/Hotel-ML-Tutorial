"""
Predictions page for the Hotel Booking Analytics application.
Provides interactive prediction interface and forecasting tools.
"""

import streamlit as st
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.components.ui_components import (
    render_section_header, render_info_box, render_learning_objectives,
    render_methodology_explanation, render_kpi_row, create_download_button
)
from src.utils.viz_utils import (
    plot_distribution, plot_time_series, plot_monthly_pattern
)
from src.config import MONTH_ORDER


def render_single_prediction():
    """Render single booking prediction interface."""
    
    render_section_header("Single Booking Prediction", "person")
    
    render_info_box(
        """Enter the details of a booking to predict whether it's likely to be canceled. 
        This simulation helps understand how different factors influence cancellation risk.""",
        title="How to Use"
    )
    
    # Check if model is available
    if 'model_comparison' not in st.session_state:
        render_info_box(
            "Please train a model first in the 'Models' section.",
            title="Model Required",
            box_type="warning"
        )
        return
    
    model = st.session_state['model_comparison'].get_best_model()
    
    st.markdown("### Booking Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Booking Information**")
        hotel = st.selectbox("Hotel Type", ['Resort Hotel', 'City Hotel'])
        lead_time = st.number_input("Lead Time (days)", 0, 500, 30)
        arrival_month = st.selectbox("Arrival Month", MONTH_ORDER)
        arrival_week = st.number_input("Week Number", 1, 53, 25)
        arrival_day = st.number_input("Day of Month", 1, 31, 15)
    
    with col2:
        st.markdown("**Stay Details**")
        weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
        week_nights = st.number_input("Week Nights", 0, 20, 2)
        adults = st.number_input("Adults", 1, 10, 2)
        children = st.number_input("Children", 0, 5, 0)
        babies = st.number_input("Babies", 0, 3, 0)
    
    with col3:
        st.markdown("**Additional Info**")
        meal = st.selectbox("Meal Plan", ['BB', 'HB', 'FB', 'SC'])
        market_segment = st.selectbox(
            "Market Segment",
            ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate', 'Groups']
        )
        deposit_type = st.selectbox(
            "Deposit Type",
            ['No Deposit', 'Non Refund', 'Refundable']
        )
        customer_type = st.selectbox(
            "Customer Type",
            ['Transient', 'Transient-Party', 'Contract', 'Group']
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Guest History**")
        is_repeated = st.checkbox("Repeat Guest")
        prev_cancellations = st.number_input("Previous Cancellations", 0, 10, 0)
        prev_bookings = st.number_input("Previous Completed Bookings", 0, 50, 0)
    
    with col2:
        st.markdown("**Other Details**")
        adr = st.number_input("Average Daily Rate ($)", 0.0, 500.0, 100.0)
        special_requests = st.number_input("Special Requests", 0, 5, 0)
        booking_changes = st.number_input("Booking Changes", 0, 10, 0)
        parking = st.number_input("Parking Spaces Required", 0, 3, 0)
        waiting_days = st.number_input("Days in Waiting List", 0, 100, 0)
    
    # Create prediction dataframe
    if st.button("Predict Cancellation Risk", type="primary"):
        # Create input data
        input_data = pd.DataFrame({
            'hotel': [hotel],
            'lead_time': [lead_time],
            'arrival_date_year': [2024],
            'arrival_date_month': [arrival_month],
            'arrival_date_week_number': [arrival_week],
            'arrival_date_day_of_month': [arrival_day],
            'stays_in_weekend_nights': [weekend_nights],
            'stays_in_week_nights': [week_nights],
            'adults': [adults],
            'children': [float(children)],
            'babies': [babies],
            'meal': [meal],
            'market_segment': [market_segment],
            'distribution_channel': ['TA/TO' if 'TA' in market_segment else 'Direct'],
            'is_repeated_guest': [1 if is_repeated else 0],
            'previous_cancellations': [prev_cancellations],
            'previous_bookings_not_canceled': [prev_bookings],
            'reserved_room_type': ['A'],
            'assigned_room_type': ['A'],
            'booking_changes': [booking_changes],
            'deposit_type': [deposit_type],
            'agent': [0.0],
            'company': [0.0],
            'days_in_waiting_list': [waiting_days],
            'customer_type': [customer_type],
            'adr': [adr],
            'required_car_parking_spaces': [parking],
            'total_of_special_requests': [special_requests],
        })
        
        try:
            prediction, probability = model.predict(input_data)
            
            st.markdown("---")
            st.markdown("### Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction[0] == 1:
                    st.error("HIGH CANCELLATION RISK")
                else:
                    st.success("LOW CANCELLATION RISK")
            
            with col2:
                st.metric("Cancellation Probability", f"{probability[0]*100:.1f}%")
            
            with col3:
                risk_level = "High" if probability[0] > 0.7 else "Medium" if probability[0] > 0.4 else "Low"
                st.metric("Risk Level", risk_level)
            
            # Risk factors analysis
            st.markdown("### Risk Factor Analysis")
            
            risk_factors = []
            if lead_time > 100:
                risk_factors.append(f"Long lead time ({lead_time} days) increases cancellation risk")
            if deposit_type == 'No Deposit':
                risk_factors.append("No deposit required - lower commitment")
            if prev_cancellations > 0:
                risk_factors.append(f"Guest has {prev_cancellations} previous cancellations")
            if not is_repeated:
                risk_factors.append("First-time guest - no loyalty established")
            if special_requests == 0:
                risk_factors.append("No special requests - less invested in the stay")
            
            protective_factors = []
            if is_repeated:
                protective_factors.append("Repeat guest - established loyalty")
            if special_requests > 0:
                protective_factors.append(f"{special_requests} special request(s) - invested in stay")
            if deposit_type != 'No Deposit':
                protective_factors.append(f"{deposit_type} deposit required")
            if prev_bookings > 0:
                protective_factors.append(f"{prev_bookings} previous completed bookings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Risk Factors:**")
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("No significant risk factors identified")
            
            with col2:
                st.markdown("**Protective Factors:**")
                if protective_factors:
                    for factor in protective_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("No protective factors identified")
            
            # Recommendations
            st.markdown("### Recommendations")
            
            if probability[0] > 0.7:
                render_info_box(
                    """**High Risk Booking - Consider:**
                    - Request a non-refundable deposit
                    - Send confirmation reminders
                    - Prepare for potential overbooking opportunity
                    - Contact guest to confirm booking closer to arrival date""",
                    box_type="warning"
                )
            elif probability[0] > 0.4:
                render_info_box(
                    """**Medium Risk Booking - Consider:**
                    - Send friendly confirmation emails
                    - Offer early check-in or other perks for confirmation
                    - Monitor for booking changes""",
                    box_type="info"
                )
            else:
                render_info_box(
                    """**Low Risk Booking:**
                    - Standard booking procedures apply
                    - Consider upselling opportunities
                    - Focus on enhancing guest experience""",
                    box_type="success"
                )
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


def render_batch_prediction(df: pd.DataFrame):
    """Render batch prediction interface."""
    
    render_section_header("Batch Predictions", "upload_file")
    
    render_info_box(
        """Upload a dataset or use the loaded data to predict cancellation risk 
        for multiple bookings at once. This is useful for analyzing upcoming 
        reservations and planning accordingly.""",
        title="Batch Processing"
    )
    
    if 'model_comparison' not in st.session_state:
        render_info_box(
            "Please train a model first in the 'Models' section.",
            title="Model Required",
            box_type="warning"
        )
        return
    
    model = st.session_state['model_comparison'].get_best_model()
    
    # Option to use sample or upload
    data_source = st.radio(
        "Select data source:",
        ["Use sample from dataset", "Upload CSV file"],
        horizontal=True
    )
    
    if data_source == "Use sample from dataset":
        sample_size = st.slider("Sample size", 100, 5000, 500, 100)
        prediction_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            prediction_df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file with booking data.")
            return
    
    st.markdown(f"**Data loaded:** {len(prediction_df):,} bookings")
    
    if st.button("Run Batch Prediction", type="primary"):
        with st.spinner("Making predictions..."):
            try:
                predictions, probabilities = model.predict(prediction_df)
                
                # Add predictions to dataframe
                result_df = prediction_df.copy()
                result_df['predicted_cancellation'] = predictions
                result_df['cancellation_probability'] = probabilities
                result_df['risk_level'] = pd.cut(
                    probabilities,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=['Low', 'Medium', 'High']
                )
                
                st.success("Predictions completed!")
                
                # Summary statistics
                st.markdown("### Prediction Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Bookings", f"{len(result_df):,}")
                with col2:
                    st.metric("Predicted Cancellations", f"{predictions.sum():,}")
                with col3:
                    st.metric("Predicted Rate", f"{predictions.mean()*100:.1f}%")
                with col4:
                    st.metric("Avg Probability", f"{probabilities.mean()*100:.1f}%")
                
                # Risk distribution
                st.markdown("### Risk Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_counts = result_df['risk_level'].value_counts()
                    st.dataframe(risk_counts.reset_index().rename(
                        columns={'index': 'Risk Level', 'risk_level': 'Count'}
                    ))
                
                with col2:
                    fig = plot_distribution(
                        result_df, 'cancellation_probability',
                        title="Distribution of Cancellation Probabilities",
                        bins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show results
                st.markdown("### Detailed Results")
                
                # Filter options
                risk_filter = st.multiselect(
                    "Filter by risk level:",
                    ['Low', 'Medium', 'High'],
                    default=['High', 'Medium']
                )
                
                filtered_df = result_df[result_df['risk_level'].isin(risk_filter)]
                
                display_cols = [
                    'hotel', 'lead_time', 'arrival_date_month', 'adr',
                    'customer_type', 'deposit_type', 'predicted_cancellation',
                    'cancellation_probability', 'risk_level'
                ]
                display_cols = [c for c in display_cols if c in filtered_df.columns]
                
                st.dataframe(
                    filtered_df[display_cols].head(100).sort_values(
                        'cancellation_probability', ascending=False
                    ),
                    use_container_width=True
                )
                
                # Download option
                create_download_button(
                    result_df,
                    "predictions_results.csv",
                    "Download Full Results"
                )
                
                # Store for later use
                st.session_state['batch_predictions'] = result_df
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")


def render_demand_forecast(df: pd.DataFrame):
    """Render demand forecasting section."""
    
    render_section_header("Demand Forecasting", "trending_up")
    
    render_methodology_explanation(
        "Time Series Forecasting",
        """Demand forecasting uses historical booking data to predict future demand. 
        This helps with capacity planning, staffing, and revenue optimization.""",
        [
            "Analyze historical booking patterns",
            "Identify seasonal trends and cycles",
            "Account for year-over-year growth",
            "Generate forecasts with confidence intervals"
        ]
    )
    
    # Prepare time series data
    ts_data = df.groupby(df['arrival_date'].dt.to_period('M')).size().reset_index()
    ts_data.columns = ['Period', 'Bookings']
    ts_data['Period'] = ts_data['Period'].astype(str)
    
    # Historical trends
    st.markdown("### Historical Booking Trends")
    
    fig = plot_time_series(
        ts_data, 'Period', 'Bookings',
        title="Monthly Booking Volume (Historical)",
        show_trend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal decomposition visualization
    st.markdown("### Seasonal Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_monthly_pattern(
            df, 'is_canceled', 'count',
            "Average Monthly Bookings"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cancellation patterns by month
        fig = plot_monthly_pattern(
            df, 'is_canceled', 'mean',
            "Average Cancellation Rate by Month"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Simple forecast using seasonal average
    st.markdown("### Demand Forecast (Next 12 Months)")
    
    render_info_box(
        """This forecast uses historical monthly averages adjusted for year-over-year 
        growth trends. For production use, consider implementing more sophisticated 
        methods like Prophet, SARIMA, or machine learning-based forecasting.""",
        title="Forecasting Methodology"
    )
    
    # Calculate monthly averages
    monthly_avg = df.groupby('arrival_date_month')['is_canceled'].count()
    monthly_avg = monthly_avg.reindex(MONTH_ORDER)
    
    # Calculate growth rate
    yearly_totals = df.groupby('arrival_date_year').size()
    if len(yearly_totals) > 1:
        growth_rate = (yearly_totals.iloc[-1] / yearly_totals.iloc[-2]) - 1
    else:
        growth_rate = 0.05  # Default 5% growth
    
    # Generate forecast
    forecast_months = MONTH_ORDER
    forecast_values = monthly_avg.values * (1 + growth_rate)
    
    forecast_df = pd.DataFrame({
        'Month': forecast_months,
        'Forecasted_Bookings': forecast_values.astype(int),
        'Lower_Bound': (forecast_values * 0.85).astype(int),
        'Upper_Bound': (forecast_values * 1.15).astype(int)
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    
    with col2:
        total_forecast = forecast_df['Forecasted_Bookings'].sum()
        st.metric("Forecasted Annual Bookings", f"{total_forecast:,}")
        st.metric("Assumed Growth Rate", f"{growth_rate*100:.1f}%")
        st.metric("Peak Month", 
                  forecast_df.loc[forecast_df['Forecasted_Bookings'].idxmax(), 'Month'])
    
    render_info_box(
        f"""**Planning Recommendations:**
        
        - **Peak Season (Summer)**: Prepare for 30-40% higher demand in July-August
        - **Staffing**: Increase staffing levels 2 months before peak
        - **Inventory**: Consider accepting more bookings in low-demand months
        - **Growth**: Plan for {growth_rate*100:.1f}% year-over-year growth""",
        title="Capacity Planning Insights"
    )


def render_what_if_analysis(df: pd.DataFrame):
    """Render what-if scenario analysis."""
    
    render_section_header("What-If Analysis", "science")
    
    render_info_box(
        """Explore different scenarios to understand how changes in booking 
        characteristics might affect cancellation rates. This helps in policy 
        development and risk assessment.""",
        title="Scenario Analysis"
    )
    
    if 'model_comparison' not in st.session_state:
        render_info_box(
            "Please train a model first in the 'Models' section.",
            title="Model Required",
            box_type="warning"
        )
        return
    
    model = st.session_state['model_comparison'].get_best_model()
    
    st.markdown("### Scenario: Impact of Lead Time on Cancellations")
    
    # Generate scenarios
    lead_times = list(range(0, 365, 30))
    base_booking = {
        'hotel': 'City Hotel',
        'arrival_date_year': 2024,
        'arrival_date_month': 'July',
        'arrival_date_week_number': 28,
        'arrival_date_day_of_month': 15,
        'stays_in_weekend_nights': 1,
        'stays_in_week_nights': 2,
        'adults': 2,
        'children': 0.0,
        'babies': 0,
        'meal': 'BB',
        'market_segment': 'Online TA',
        'distribution_channel': 'TA/TO',
        'is_repeated_guest': 0,
        'previous_cancellations': 0,
        'previous_bookings_not_canceled': 0,
        'reserved_room_type': 'A',
        'assigned_room_type': 'A',
        'booking_changes': 0,
        'deposit_type': 'No Deposit',
        'agent': 0.0,
        'company': 0.0,
        'days_in_waiting_list': 0,
        'customer_type': 'Transient',
        'adr': 100.0,
        'required_car_parking_spaces': 0,
        'total_of_special_requests': 0,
    }
    
    scenarios = []
    for lt in lead_times:
        booking = base_booking.copy()
        booking['lead_time'] = lt
        scenarios.append(booking)
    
    scenario_df = pd.DataFrame(scenarios)
    
    try:
        _, probabilities = model.predict(scenario_df)
        
        result_df = pd.DataFrame({
            'Lead Time (days)': lead_times,
            'Cancellation Probability': probabilities
        })
        
        fig = plot_time_series(
            result_df, 'Lead Time (days)', 'Cancellation Probability',
            title="Cancellation Probability vs Lead Time"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        render_info_box(
            """This analysis shows how cancellation probability increases with longer 
            lead times. Hotels can use this insight to:
            - Implement stricter deposit policies for bookings made far in advance
            - Send more frequent confirmation reminders for long lead time bookings
            - Adjust overbooking strategies based on lead time distribution""",
            title="Key Insight"
        )
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Scenario: Impact of Deposit Policy")
    
    deposit_scenarios = []
    for deposit in ['No Deposit', 'Non Refund', 'Refundable']:
        booking = base_booking.copy()
        booking['deposit_type'] = deposit
        booking['lead_time'] = 60
        deposit_scenarios.append(booking)
    
    deposit_df = pd.DataFrame(deposit_scenarios)
    
    try:
        _, probs = model.predict(deposit_df)
        
        deposit_results = pd.DataFrame({
            'Deposit Type': ['No Deposit', 'Non Refund', 'Refundable'],
            'Cancellation Probability': probs * 100
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(deposit_results, use_container_width=True, hide_index=True)
        
        with col2:
            render_info_box(
                """Deposit policies significantly impact cancellation behavior:
                - No Deposit: Highest cancellation risk
                - Non-Refundable: Strong deterrent to cancellation
                - Refundable: Moderate protection""",
                title="Policy Impact"
            )
            
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")


def render_page(df: pd.DataFrame):
    """Main render function for the Predictions page."""
    
    render_learning_objectives("predictions")
    
    tabs = st.tabs([
        "Single Prediction",
        "Batch Predictions",
        "Demand Forecast",
        "What-If Analysis"
    ])
    
    with tabs[0]:
        render_single_prediction()
    
    with tabs[1]:
        render_batch_prediction(df)
    
    with tabs[2]:
        render_demand_forecast(df)
    
    with tabs[3]:
        render_what_if_analysis(df)
