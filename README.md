# Hotel Booking Analytics - Data Science Learning Platform

A comprehensive, interactive data science learning platform built with Streamlit. This application guides users through the complete data science workflow using real hotel booking data, from exploratory data analysis to machine learning model deployment and predictions.

## Features

### 1. Data Understanding
- Dataset overview with key statistics
- Comprehensive data dictionary
- Data quality assessment and missing value analysis
- Statistical summaries for numerical and categorical features
- Business context and key questions to answer

### 2. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Distribution analysis for numerical and categorical variables
- **Bivariate Analysis**: Correlation analysis and feature relationships
- **Cancellation Patterns**: In-depth analysis of cancellation factors
- **Temporal Patterns**: Seasonal and time-based trend analysis
- **Revenue Analysis**: ADR patterns and revenue insights
- **Guest Behavior**: Customer segmentation and behavior analysis

### 3. Machine Learning Models
- **Algorithm Education**: Detailed explanations of each ML algorithm
- **Model Training**: Train and compare multiple classifiers:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Decision Tree
  - K-Nearest Neighbors
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1, ROC-AUC
- **Feature Importance**: Understand which factors most influence predictions
- **Business Impact Analysis**: Translate model performance into business value

### 4. Predictions & Forecasting
- **Single Booking Prediction**: Interactive form for individual predictions
- **Batch Predictions**: Process multiple bookings at once
- **Demand Forecasting**: Historical trend analysis and future demand prediction
- **What-If Analysis**: Scenario exploration for policy decisions

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mohamadsolouki/Hotel-ML-Tutorial.git
cd Hotel-ML-Tutorial
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
Hotel-ML-Tutorial/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── hotel_bookings.xlsx         # Dataset
├── README.md                   # This file
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── assets/                     # Static assets (icons, images)
└── src/
    ├── __init__.py
    ├── config.py              # Configuration and constants
    ├── components/
    │   ├── __init__.py
    │   └── ui_components.py   # Reusable UI components
    ├── models/
    │   ├── __init__.py
    │   └── ml_models.py       # Machine learning models
    ├── pages/
    │   ├── __init__.py
    │   ├── data_understanding.py
    │   ├── eda.py
    │   ├── models.py
    │   └── predictions.py
    └── utils/
        ├── __init__.py
        ├── data_utils.py      # Data loading and preprocessing
        └── viz_utils.py       # Visualization utilities
```

## Dataset

The application uses the Hotel Booking Demand dataset containing **119,390 hotel bookings** with **32 features** including:

- **Booking Details**: Lead time, arrival dates, stay duration
- **Guest Information**: Adults, children, country of origin
- **Booking Characteristics**: Market segment, distribution channel, meal plan
- **Financial**: Average Daily Rate (ADR)
- **Outcome**: Cancellation status (target variable)

### Key Statistics
- **Date Range**: July 2015 - August 2017
- **Hotel Types**: Resort Hotel, City Hotel
- **Cancellation Rate**: ~37%
- **Countries**: 178 unique countries

## Key Business Questions Addressed

### Seasonal Booking Patterns
- How do hotel booking volumes vary across months and years?
- What seasonal patterns can be identified in booking behavior?
- Are there differences in seasonality between resort and city hotels?

### Demand Forecasting
- Can historical booking trends forecast future demand?
- What factors most influence booking demand?
- What is the optimal planning horizon for capacity management?

### Cancellation Analysis
- How do cancellation rates fluctuate over time?
- What factors are most predictive of cancellations?
- Which customer segments have the highest cancellation rates?

### Revenue Management
- How does Average Daily Rate (ADR) vary by season?
- What is the relationship between ADR and cancellation?
- Which market segments generate the highest revenue?

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Statistical Analysis**: StatsModels

## Usage Guide

### Getting Started
1. Launch the application and explore the **Home** page for an overview
2. Navigate through sections using the sidebar
3. Each section includes educational content explaining the methodology

### Training Models
1. Go to **Machine Learning** > **Training** tab
2. Select models to train
3. Adjust sample size for faster training during exploration
4. Click "Train Models" to begin

### Making Predictions
1. First train a model in the **Machine Learning** section
2. Go to **Predictions** section
3. Use single prediction for individual bookings
4. Use batch prediction for multiple bookings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: Hotel Booking Demand dataset
- Built with Streamlit framework
- Inspired by the need for practical, interactive data science education