"""
Configuration settings for the Hotel Booking Analytics application.
Contains all constants, color schemes, and configuration parameters.
"""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATIONS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "hotel_bookings.xlsx"
ASSETS_PATH = PROJECT_ROOT / "assets"

# =============================================================================
# COLOR SCHEME - Professional Data Science Theme
# =============================================================================
COLORS = {
    "primary": "#1E3A5F",      # Deep Navy Blue
    "secondary": "#3D5A80",    # Steel Blue
    "accent": "#EE6C4D",       # Coral Orange
    "success": "#2ECC71",      # Emerald Green
    "warning": "#F39C12",      # Golden Yellow
    "danger": "#E74C3C",       # Red
    "background": "#F8F9FA",   # Light Gray
    "card_bg": "#FFFFFF",      # White
    "text_primary": "#2C3E50", # Dark Slate
    "text_secondary": "#7F8C8D", # Gray
    "gradient_start": "#667eea",
    "gradient_end": "#764ba2",
}

# Plotly color palette for charts
PLOTLY_COLORS = [
    "#1E3A5F", "#3D5A80", "#EE6C4D", "#98C1D9", "#293241",
    "#E0FBFC", "#F07167", "#00B4D8", "#90E0EF", "#CAF0F8"
]

# Sequential color scale for heatmaps
SEQUENTIAL_COLORS = [
    [0, "#F8F9FA"],
    [0.5, "#3D5A80"],
    [1, "#1E3A5F"]
]

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
CATEGORICAL_COLUMNS = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'customer_type', 'reservation_status'
]

NUMERICAL_COLUMNS = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests'
]

TARGET_COLUMN = 'is_canceled'

MONTH_ORDER = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature sets for different model types
FEATURES_FOR_CANCELLATION = [
    'lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
    'babies', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
    'adr', 'required_car_parking_spaces', 'total_of_special_requests'
]

# =============================================================================
# UI CONFIGURATION
# =============================================================================
PAGE_CONFIG = {
    "page_title": "Hotel Booking Analytics",
    "page_icon": "assets/hotel_icon.png",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Section icons (Material Design Icons)
ICONS = {
    "overview": "dashboard",
    "data": "database",
    "eda": "bar_chart",
    "analysis": "analytics",
    "models": "model_training",
    "predictions": "insights",
    "info": "info",
    "success": "check_circle",
    "warning": "warning",
    "error": "error",
    "calendar": "calendar_month",
    "time": "schedule",
    "money": "payments",
    "people": "groups",
    "hotel": "hotel",
    "trend": "trending_up",
    "download": "download",
    "settings": "settings",
}

# =============================================================================
# EDUCATIONAL CONTENT
# =============================================================================
LEARNING_OBJECTIVES = {
    "overview": [
        "Understand the hotel booking dataset structure and contents",
        "Identify key business questions that can be answered with the data",
        "Learn about the importance of data exploration before modeling"
    ],
    "eda": [
        "Perform univariate and bivariate analysis on hotel booking data",
        "Identify patterns, outliers, and anomalies in the dataset",
        "Understand distributions of key variables",
        "Visualize relationships between features and the target variable"
    ],
    "analysis": [
        "Analyze seasonal booking patterns and trends",
        "Understand factors influencing booking cancellations",
        "Examine customer segmentation and behavior patterns",
        "Perform revenue and demand analysis"
    ],
    "models": [
        "Understand model selection for classification problems",
        "Learn about feature engineering and preprocessing",
        "Compare multiple machine learning algorithms",
        "Evaluate model performance using various metrics"
    ],
    "predictions": [
        "Apply trained models to make predictions",
        "Understand prediction confidence and uncertainty",
        "Use predictions for business decision making",
        "Learn about model deployment considerations"
    ]
}

# Business questions organized by category
BUSINESS_QUESTIONS = {
    "Seasonal Patterns": [
        "How do hotel booking volumes vary across months and years?",
        "What seasonal patterns can be identified in booking behavior?",
        "Are there differences in seasonality between resort and city hotels?",
        "How does lead time vary by season?"
    ],
    "Demand Forecasting": [
        "Can historical booking trends forecast future demand?",
        "What factors most influence booking demand?",
        "How accurate can demand predictions be?",
        "What is the optimal planning horizon for capacity management?"
    ],
    "Cancellation Analysis": [
        "How do cancellation rates fluctuate over time?",
        "What factors are most predictive of cancellations?",
        "Which customer segments have highest cancellation rates?",
        "How does lead time affect cancellation probability?"
    ],
    "Revenue Management": [
        "How does Average Daily Rate (ADR) vary by season?",
        "What is the relationship between ADR and cancellation?",
        "Which market segments generate highest revenue?",
        "How can pricing strategies be optimized?"
    ],
    "Customer Behavior": [
        "What are the characteristics of repeat guests?",
        "How do booking patterns differ by customer type?",
        "What special requests are most common?",
        "Which countries generate the most bookings?"
    ]
}
