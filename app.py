"""
Hotel Booking Analytics - A Comprehensive Data Science Learning Platform

This Streamlit application provides an interactive learning experience for data science
using real hotel booking data. It covers the complete data science workflow from
exploratory data analysis to machine learning model deployment.

Author: Hotel Analytics Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_utils import load_data
from src.components.ui_components import apply_custom_css, render_header
from src.pages import data_understanding, eda, models, predictions


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Hotel Booking Analytics",
    page_icon="assets/hotel_icon.png" if Path("assets/hotel_icon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/mohamadsolouki/Hotel-ML-Tutorial',
        'Report a bug': 'https://github.com/mohamadsolouki/Hotel-ML-Tutorial/issues',
        'About': """
        # Hotel Booking Analytics
        
        A comprehensive data science learning platform using hotel booking data.
        
        This application demonstrates:
        - Exploratory Data Analysis (EDA)
        - Statistical Analysis
        - Machine Learning Modeling
        - Prediction & Forecasting
        
        Built with Streamlit, Pandas, Scikit-learn, and Plotly.
        """
    }
)

# Apply custom styling
apply_custom_css()


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(show_spinner=False)
def get_data():
    """Load and cache the hotel booking data."""
    return load_data()


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    """Render the sidebar navigation and information."""
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #1E3A5F; margin: 0;">Hotel Analytics</h1>
            <p style="color: #7F8C8D; font-size: 0.9rem;">Data Science Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        
        pages = {
            "Home": "home",
            "1. Data Understanding": "data",
            "2. Exploratory Analysis": "eda",
            "3. Machine Learning": "models",
            "4. Predictions": "predictions"
        }
        
        # Use session state for page selection
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "home"
        
        for page_name, page_key in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #7F8C8D; font-size: 0.8rem;">
            <p>Built with Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
    
    return st.session_state.current_page


# =============================================================================
# HOME PAGE
# =============================================================================
def render_home_page(df: pd.DataFrame):
    """Render the home/landing page."""
    
    render_header(
        "Hotel Booking Analytics",
        "A Comprehensive Data Science Learning Platform"
    )
    
    # Introduction
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
        <h2 style="color: #1E3A5F; margin-top: 0;">Welcome to the Hotel Booking Analytics Platform</h2>
        <p style="font-size: 1.1rem; color: #2C3E50;">
            This interactive application guides you through a complete data science workflow using 
            real hotel booking data. Learn how to analyze booking patterns, build machine learning 
            models, and make predictions that can inform business decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1E3A5F 0%, #3D5A80 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700;">{:,}</div>
            <div style="opacity: 0.9;">Total Bookings</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #EE6C4D 0%, #F39C12 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700;">{:.1f}%</div>
            <div style="opacity: 0.9;">Cancellation Rate</div>
        </div>
        """.format(df['is_canceled'].mean()*100), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700;">{}</div>
            <div style="opacity: 0.9;">Features</div>
        </div>
        """.format(len(df.columns)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #9B59B6 0%, #8E44AD 100%); 
                    padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700;">{}</div>
            <div style="opacity: 0.9;">Countries</div>
        </div>
        """.format(df['country'].nunique()), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Learning modules
    st.markdown("## Learning Modules")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid #1E3A5F; margin-bottom: 1rem;">
            <h3 style="color: #1E3A5F; margin-top: 0;">
                <span class="material-symbols-outlined" style="vertical-align: middle;">database</span>
                1. Data Understanding
            </h3>
            <p style="color: #2C3E50;">
                Explore the dataset structure, understand each feature, assess data quality, 
                and identify the business questions we can answer.
            </p>
            <ul style="color: #7F8C8D;">
                <li>Dataset overview and statistics</li>
                <li>Data dictionary and feature descriptions</li>
                <li>Missing value analysis</li>
                <li>Business context and questions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid #2ECC71; margin-bottom: 1rem;">
            <h3 style="color: #1E3A5F; margin-top: 0;">
                <span class="material-symbols-outlined" style="vertical-align: middle;">model_training</span>
                3. Machine Learning
            </h3>
            <p style="color: #2C3E50;">
                Build and compare multiple ML models to predict booking cancellations. 
                Understand each algorithm and evaluate their performance.
            </p>
            <ul style="color: #7F8C8D;">
                <li>Algorithm explanations</li>
                <li>Model training and comparison</li>
                <li>Performance evaluation</li>
                <li>Feature importance analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid #EE6C4D; margin-bottom: 1rem;">
            <h3 style="color: #1E3A5F; margin-top: 0;">
                <span class="material-symbols-outlined" style="vertical-align: middle;">bar_chart</span>
                2. Exploratory Analysis
            </h3>
            <p style="color: #2C3E50;">
                Dive deep into the data through visualizations and statistical analysis. 
                Discover patterns, trends, and relationships.
            </p>
            <ul style="color: #7F8C8D;">
                <li>Univariate and bivariate analysis</li>
                <li>Cancellation pattern analysis</li>
                <li>Temporal and seasonal trends</li>
                <li>Revenue and guest analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                    border-left: 4px solid #9B59B6; margin-bottom: 1rem;">
            <h3 style="color: #1E3A5F; margin-top: 0;">
                <span class="material-symbols-outlined" style="vertical-align: middle;">insights</span>
                4. Predictions
            </h3>
            <p style="color: #2C3E50;">
                Apply trained models to make predictions. Explore demand forecasting 
                and scenario analysis for business planning.
            </p>
            <ul style="color: #7F8C8D;">
                <li>Single booking prediction</li>
                <li>Batch predictions</li>
                <li>Demand forecasting</li>
                <li>What-if scenario analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Key questions
    st.markdown("## Key Business Questions")
    
    questions = [
        ("How do hotel booking volumes vary across months and years?", 
         "Analyze seasonal patterns and trends in booking behavior."),
        ("Can historical booking trends forecast future demand?", 
         "Build predictive models for capacity planning."),
        ("How do cancellation rates fluctuate over time?", 
         "Understand patterns for revenue management."),
        ("What factors most influence booking cancellations?", 
         "Identify key predictors to reduce cancellations."),
        ("Which customer segments have the highest cancellation rates?", 
         "Target high-risk segments with retention strategies."),
        ("How does lead time affect cancellation probability?", 
         "Inform deposit and overbooking policies."),
    ]
    
    col1, col2 = st.columns(2)
    
    for i, (question, description) in enumerate(questions):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                <strong style="color: #1E3A5F;">{question}</strong>
                <p style="color: #7F8C8D; margin: 0.5rem 0 0 0; font-size: 0.9rem;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Getting started
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E3A5F 0%, #3D5A80 100%); 
                padding: 2rem; border-radius: 12px; text-align: center; color: white;">
        <h2 style="margin-top: 0;">Ready to Start?</h2>
        <p style="opacity: 0.9; max-width: 600px; margin: 0 auto 1rem auto;">
            Navigate through the sections using the sidebar to explore the data, 
            build models, and make predictions. Each section includes detailed 
            explanations and interactive visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    
    # Load data
    try:
        df = get_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the hotel_bookings.xlsx file is in the project root directory.")
        return
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    
    # Route to appropriate page
    if current_page == "home":
        render_home_page(df)
    elif current_page == "data":
        render_header("Data Understanding", "Explore and understand the hotel booking dataset")
        data_understanding.render_page(df)
        st.session_state['completed_data'] = True
    elif current_page == "eda":
        render_header("Exploratory Data Analysis", "Discover patterns and insights in the data")
        eda.render_page(df)
        st.session_state['completed_eda'] = True
    elif current_page == "models":
        render_header("Machine Learning Models", "Build and evaluate predictive models")
        models.render_page(df)
    elif current_page == "predictions":
        render_header("Predictions & Forecasting", "Apply models for predictions and planning")
        predictions.render_page(df)


if __name__ == "__main__":
    main()
