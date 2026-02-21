"""
Machine Learning Models page for the Hotel Booking Analytics application.
Provides model training, comparison, and evaluation functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.components.ui_components import (
    render_section_header, render_info_box, render_learning_objectives,
    render_methodology_explanation, render_kpi_row, render_progress_stepper,
    render_code_block
)
from src.utils.viz_utils import (
    plot_feature_importance, plot_confusion_matrix, plot_roc_curve
)
from src.models.ml_models import (
    CancellationPredictionModel, ModelComparison, 
    get_model_explanation, calculate_business_metrics
)
from src.config import RANDOM_STATE, TEST_SIZE, CV_FOLDS


def render_model_overview():
    """Render model overview and theory section."""
    
    render_section_header("Machine Learning Overview", "psychology")
    
    render_learning_objectives("models")
    
    st.markdown("---")
    
    render_info_box(
        """In this section, we will build machine learning models to predict hotel booking 
        cancellations. This is a **binary classification** problem where we predict whether 
        a booking will be canceled (1) or not (0). Accurate predictions can help hotels:
        
        - Optimize overbooking strategies
        - Improve revenue management
        - Enhance customer retention efforts
        - Better allocate resources""",
        title="Problem Definition"
    )
    
    # ML Pipeline overview
    st.markdown("### Machine Learning Pipeline")
    
    render_progress_stepper([
        "Data Preprocessing",
        "Feature Engineering",
        "Model Selection",
        "Model Training",
        "Model Evaluation",
        "Deployment & Predictions"
    ], current_step=0)
    
    # Model types explanation
    st.markdown("### Available Models")
    
    model_types = ['logistic_regression', 'random_forest', 'gradient_boosting', 
                   'decision_tree', 'knn']
    
    for model_type in model_types:
        explanation = get_model_explanation(model_type)
        
        with st.expander(f"{model_type.replace('_', ' ').title()}", expanded=False):
            st.markdown(explanation['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Pros:**")
                for pro in explanation['pros']:
                    st.markdown(f"- {pro}")
            
            with col2:
                st.markdown("**Cons:**")
                for con in explanation['cons']:
                    st.markdown(f"- {con}")
            
            render_info_box(explanation['use_case'], title="Best Use Case")


def render_preprocessing_explanation(df: pd.DataFrame):
    """Render data preprocessing explanation."""
    
    render_section_header("Data Preprocessing", "build")
    
    render_methodology_explanation(
        "Feature Engineering & Preprocessing",
        """Before training models, we need to prepare our data. This involves handling 
        missing values, encoding categorical variables, and scaling numerical features.""",
        [
            "Handle missing values (imputation or removal)",
            "Encode categorical variables (Label Encoding)",
            "Scale numerical features (StandardScaler)",
            "Split data into training and test sets"
        ]
    )
    
    # Show preprocessing code
    st.markdown("#### Preprocessing Steps Code")
    
    render_code_block("""
# 1. Handle missing values
df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

# 2. Encode categorical features
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['hotel_encoded'] = label_encoder.fit_transform(df['hotel'])

# 3. Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
""", language="python")
    
    # Show feature information
    st.markdown("#### Features Used for Modeling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numerical Features:**")
        numerical_features = [
            'lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
            'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
            'babies', 'is_repeated_guest', 'previous_cancellations',
            'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
            'adr', 'required_car_parking_spaces', 'total_of_special_requests'
        ]
        for feat in numerical_features:
            st.markdown(f"- `{feat}`")
    
    with col2:
        st.markdown("**Categorical Features (to be encoded):**")
        categorical_features = [
            'hotel', 'meal', 'market_segment', 'distribution_channel',
            'reserved_room_type', 'deposit_type', 'customer_type'
        ]
        for feat in categorical_features:
            st.markdown(f"- `{feat}`")
    
    # Class balance
    st.markdown("#### Target Variable Distribution")
    
    class_dist = df['is_canceled'].value_counts()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Not Canceled (0)", f"{class_dist[0]:,}")
    with col2:
        st.metric("Canceled (1)", f"{class_dist[1]:,}")
    with col3:
        ratio = class_dist[0] / class_dist[1]
        st.metric("Class Ratio", f"{ratio:.2f}:1")
    
    render_info_box(
        f"""The dataset has a {class_dist[1]/len(df)*100:.1f}% cancellation rate, 
        which means the classes are imbalanced but not severely. For highly imbalanced 
        datasets, techniques like SMOTE, class weighting, or threshold adjustment 
        may be necessary. Our current ratio is manageable with stratified sampling.""",
        title="Class Imbalance Consideration"
    )


def render_model_training(df: pd.DataFrame):
    """Render model training section."""
    
    render_section_header("Model Training & Comparison", "model_training")
    
    st.markdown("""
    In this section, you can train multiple machine learning models and compare their 
    performance. Select the models you want to train and click the button to start.
    """)
    
    # Model selection
    st.markdown("#### Select Models to Train")
    
    col1, col2 = st.columns(2)
    
    available_models = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'decision_tree': 'Decision Tree',
        'knn': 'K-Nearest Neighbors'
    }
    
    with col1:
        selected_models = []
        for model_key, model_name in list(available_models.items())[:3]:
            if st.checkbox(model_name, value=True, key=f"model_{model_key}"):
                selected_models.append(model_key)
    
    with col2:
        for model_key, model_name in list(available_models.items())[3:]:
            if st.checkbox(model_name, value=True, key=f"model_{model_key}"):
                selected_models.append(model_key)
    
    # Training parameters
    st.markdown("#### Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    with col3:
        sample_size = st.slider("Sample Size (%)", 10, 100, 50, 10)
    
    render_info_box(
        f"""**Configuration:**
        - Test set: {test_size*100:.0f}% of data held out for evaluation
        - Cross-validation: {cv_folds}-fold CV on training data
        - Sample: Using {sample_size}% of data ({int(len(df)*sample_size/100):,} records)
        
        Note: Using a sample for faster training. Use 100% for final model.""",
        title="Training Configuration"
    )
    
    # Train button
    if st.button("Train Models", type="primary", disabled=len(selected_models) == 0):
        if len(selected_models) == 0:
            st.error("Please select at least one model to train.")
            return
        
        # Sample data
        df_sample = df.sample(frac=sample_size/100, random_state=RANDOM_STATE)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train models
        comparison = ModelComparison(selected_models)
        
        status_text.text("Training models... This may take a moment.")
        
        try:
            results_df = comparison.run_comparison(df_sample)
            progress_bar.progress(100)
            status_text.text("Training complete!")
            
            # Store results in session state
            st.session_state['model_comparison'] = comparison
            st.session_state['model_results'] = results_df
            st.session_state['trained_df'] = df_sample
            
            st.success("All models trained successfully!")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            return
    
    # Display results if available
    if 'model_results' in st.session_state:
        st.markdown("---")
        render_model_results()


def render_model_results():
    """Render model comparison results."""
    
    render_section_header("Model Comparison Results", "leaderboard")
    
    results_df = st.session_state['model_results']
    comparison = st.session_state['model_comparison']
    
    # Results table
    st.markdown("#### Performance Metrics")
    
    # Format the dataframe for display
    display_df = results_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'CV Accuracy (mean)']:
        if col in display_df.columns:
            if display_df[col].dtype != 'object':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Best model highlight
    best_model = comparison.get_best_model()
    render_info_box(
        f"""**Best Model:** {best_model.model_name}
        
        Selected based on F1 Score, which balances precision and recall. 
        This is important for cancellation prediction where both false positives 
        (predicting cancellation when guest shows up) and false negatives 
        (missing a cancellation) have business costs.""",
        title="Best Performing Model",
        box_type="success"
    )
    
    # Metrics explanation
    with st.expander("Understanding the Metrics", expanded=False):
        st.markdown("""
        | Metric | Description | Importance for Cancellation Prediction |
        |--------|-------------|---------------------------------------|
        | **Accuracy** | Overall correct predictions | General performance indicator |
        | **Precision** | Of predicted cancellations, how many were correct | Important to avoid overbooking |
        | **Recall** | Of actual cancellations, how many did we catch | Important to not miss cancellations |
        | **F1 Score** | Harmonic mean of Precision and Recall | Balanced metric for imbalanced classes |
        | **ROC AUC** | Area under ROC curve | Model's ability to distinguish classes |
        | **CV Accuracy** | Cross-validated accuracy | More robust accuracy estimate |
        """)


def render_model_evaluation(df: pd.DataFrame):
    """Render detailed model evaluation section."""
    
    render_section_header("Model Evaluation", "assessment")
    
    if 'model_comparison' not in st.session_state:
        render_info_box(
            "Please train models first in the 'Model Training' tab.",
            title="No Models Trained",
            box_type="warning"
        )
        return
    
    comparison = st.session_state['model_comparison']
    
    # Select model to evaluate
    model_names = {m: comparison.models[m].model_name for m in comparison.models.keys()}
    selected_model_key = st.selectbox(
        "Select model to evaluate:",
        list(model_names.keys()),
        format_func=lambda x: model_names[x]
    )
    
    model = comparison.get_model(selected_model_key)
    
    if model is None or not model.is_fitted:
        st.error("Selected model is not available.")
        return
    
    # Display evaluation metrics
    st.markdown("### Evaluation Results")
    
    metrics = model.metrics
    
    # KPI row
    kpi_metrics = [
        {'label': 'Accuracy', 'value': f"{metrics['accuracy']:.2%}"},
        {'label': 'Precision', 'value': f"{metrics['precision']:.2%}"},
        {'label': 'Recall', 'value': f"{metrics['recall']:.2%}"},
        {'label': 'F1 Score', 'value': f"{metrics['f1_score']:.2%}"},
    ]
    render_kpi_row(kpi_metrics)
    
    st.markdown("---")
    
    # Confusion Matrix and ROC Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Confusion Matrix")
        fig = plot_confusion_matrix(metrics['confusion_matrix'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Explain confusion matrix
        cm = metrics['confusion_matrix']
        render_info_box(
            f"""**Interpreting the Confusion Matrix:**
            - True Negatives (top-left): {cm[0][0]:,} correctly predicted non-cancellations
            - False Positives (top-right): {cm[0][1]:,} wrongly predicted as cancellations
            - False Negatives (bottom-left): {cm[1][0]:,} missed cancellations
            - True Positives (bottom-right): {cm[1][1]:,} correctly predicted cancellations"""
        )
    
    with col2:
        st.markdown("#### ROC Curve")
        if 'roc_auc' in metrics:
            fig = plot_roc_curve(
                metrics['fpr'], metrics['tpr'], metrics['roc_auc']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            render_info_box(
                f"""**ROC AUC Score: {metrics['roc_auc']:.4f}**
                
                The ROC curve plots True Positive Rate vs False Positive Rate at various 
                threshold settings. AUC closer to 1.0 indicates better discrimination 
                between classes. A random classifier would have AUC = 0.5."""
            )
        else:
            st.info("ROC curve not available for this model type.")
    
    # Feature Importance
    st.markdown("### Feature Importance")
    
    feature_importance = model.get_feature_importance()
    
    if feature_importance:
        fig = plot_feature_importance(
            list(feature_importance.keys()),
            list(feature_importance.values()),
            title="Feature Importance for Cancellation Prediction"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_features[:5]
        
        render_info_box(
            f"""**Top 5 Most Important Features:**
            1. {top_5[0][0]} ({top_5[0][1]:.4f})
            2. {top_5[1][0]} ({top_5[1][1]:.4f})
            3. {top_5[2][0]} ({top_5[2][1]:.4f})
            4. {top_5[3][0]} ({top_5[3][1]:.4f})
            5. {top_5[4][0]} ({top_5[4][1]:.4f})
            
            These features have the most influence on the model's predictions.""",
            title="Key Predictors"
        )
    else:
        render_info_box(
            "Feature importance is not available for this model type.",
            box_type="warning"
        )
    
    # Classification Report
    st.markdown("### Detailed Classification Report")
    
    report = model.get_classification_report()
    st.code(report, language="text")


def render_business_impact(df: pd.DataFrame):
    """Render business impact analysis."""
    
    render_section_header("Business Impact Analysis", "trending_up")
    
    if 'model_comparison' not in st.session_state:
        render_info_box(
            "Please train models first in the 'Model Training' tab.",
            title="No Models Trained",
            box_type="warning"
        )
        return
    
    comparison = st.session_state['model_comparison']
    best_model = comparison.get_best_model()
    trained_df = st.session_state.get('trained_df', df)
    
    render_info_box(
        """Understanding the business impact of model predictions is crucial for 
        decision-making. This analysis translates model performance into potential 
        financial and operational outcomes.""",
        title="Why Business Impact Matters"
    )
    
    st.markdown("### Financial Impact Simulation")
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        avg_booking_value = st.number_input(
            "Average Booking Value ($)",
            min_value=50, max_value=1000, value=150, step=10
        )
    
    with col2:
        overbooking_cost_pct = st.slider(
            "Overbooking Cost (% of booking value)",
            10, 100, 50, 5
        )
    
    # Calculate impact
    predictions, probabilities = best_model.predict(trained_df)
    business_metrics = calculate_business_metrics(
        trained_df, predictions, probabilities, avg_booking_value
    )
    
    # Display metrics
    st.markdown("### Prediction Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted Cancellations", f"{business_metrics['predicted_cancellations']:,}")
    with col2:
        st.metric("Actual Cancellations", f"{business_metrics['actual_cancellations']:,}")
    with col3:
        st.metric("Correctly Predicted", f"{business_metrics['correctly_predicted']:,}")
    with col4:
        st.metric("Missed Cancellations", f"{business_metrics['missed_cancellations']:,}")
    
    st.markdown("### Financial Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Potential Savings",
            f"${business_metrics['potential_savings']:,.0f}",
            help="Revenue protected by correctly predicting cancellations"
        )
    
    with col2:
        st.metric(
            "Overbooking Risk",
            f"${business_metrics['potential_overbooking_loss']:,.0f}",
            delta=f"-{business_metrics['false_alarms']} false alarms",
            delta_color="inverse"
        )
    
    with col3:
        net = business_metrics['net_benefit']
        st.metric(
            "Net Benefit",
            f"${net:,.0f}",
            delta="positive" if net > 0 else "negative"
        )
    
    render_info_box(
        f"""**How to Interpret:**
        
        - **Potential Savings**: By correctly identifying {business_metrics['correctly_predicted']:,} 
          cancellations, the hotel can proactively resell rooms or adjust staffing.
        
        - **Overbooking Risk**: {business_metrics['false_alarms']:,} false positives could lead 
          to overbooking issues if the hotel accepts additional reservations.
        
        - **Missed Cancellations**: {business_metrics['missed_cancellations']:,} cancellations 
          were not predicted, potentially leading to lost revenue opportunities.
        
        - **Average Cancellation Probability**: {business_metrics['avg_cancellation_probability']:.1%}""",
        title="Business Interpretation"
    )
    
    # Recommendations
    st.markdown("### Recommendations")
    
    render_info_box(
        """Based on the model analysis, consider these strategies:
        
        1. **Early Intervention**: Contact guests with high cancellation probability 
           (>70%) to confirm their booking or offer incentives to keep it.
        
        2. **Deposit Policies**: Require deposits for bookings with high cancellation risk, 
           especially those with long lead times.
        
        3. **Overbooking Strategy**: Use prediction probabilities to inform controlled 
           overbooking decisions, balancing revenue optimization with guest experience.
        
        4. **Marketing**: Target predicted cancellations with remarketing campaigns 
           or alternative date offers.""",
        title="Actionable Strategies",
        box_type="success"
    )


def render_page(df: pd.DataFrame):
    """Main render function for the Models page."""
    
    tabs = st.tabs([
        "Overview",
        "Preprocessing",
        "Training",
        "Evaluation",
        "Business Impact"
    ])
    
    with tabs[0]:
        render_model_overview()
    
    with tabs[1]:
        render_preprocessing_explanation(df)
    
    with tabs[2]:
        render_model_training(df)
    
    with tabs[3]:
        render_model_evaluation(df)
    
    with tabs[4]:
        render_business_impact(df)
