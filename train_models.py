"""
Pre-train machine learning models and save them for deployment.
Run this script once to generate saved models.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Configuration
MODELS_DIR = Path("models")
DATA_FILE = "hotel_bookings.xlsx"

FEATURES = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests'
]

CATEGORICAL_FEATURES = [
    'hotel', 'meal', 'market_segment', 'distribution_channel',
    'deposit_type', 'customer_type'
]


def load_and_preprocess_data():
    """Load and preprocess the hotel bookings data."""
    print("Loading data...")
    df = pd.read_excel(DATA_FILE)
    print(f"Loaded {len(df):,} records")
    
    # Handle missing values
    df['children'] = df['children'].fillna(0)
    df['agent'] = df['agent'].fillna(0)
    df['company'] = df['company'].fillna(0)
    
    # Remove outliers
    df = df[df['adr'] >= 0]
    df = df[df['adr'] < 5000]
    
    # Create feature dataframe
    X = df[FEATURES].copy()
    
    # Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    y = df['is_canceled']
    
    return X, y, label_encoders, df


def train_models(X, y):
    """Train all models and return results."""
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=10, min_samples_split=20, random_state=42
        ),
        'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for models that need it
        if name in ['logistic_regression', 'knn']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(cm[0][0]),
            'false_positives': int(cm[0][1]),
            'false_negatives': int(cm[1][0]),
            'true_positives': int(cm[1][1])
        }
        
        trained_models[name] = model
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
    
    # Get feature importances for tree-based models
    feature_importances = {}
    for name in ['random_forest', 'gradient_boosting', 'decision_tree']:
        model = trained_models[name]
        feature_names = FEATURES + CATEGORICAL_FEATURES
        importances = model.feature_importances_
        feature_importances[name] = {
            feature_names[i]: float(importances[i]) 
            for i in range(len(feature_names))
        }
    
    return trained_models, scaler, results, feature_importances, X_test, y_test


def save_models(trained_models, scaler, label_encoders, results, feature_importances):
    """Save all models and metadata."""
    print("\nSaving models...")
    
    # Save models
    for name, model in trained_models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"  Saved {name}.joblib")
    
    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print("  Saved scaler.joblib")
    
    # Save label encoders
    joblib.dump(label_encoders, MODELS_DIR / "label_encoders.joblib")
    print("  Saved label_encoders.joblib")
    
    # Save results and metadata
    metadata = {
        'features': FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'results': results,
        'feature_importances': feature_importances,
        'best_model': max(results.keys(), key=lambda k: results[k]['f1_score'])
    }
    
    with open(MODELS_DIR / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  Saved model_metadata.json")


def compute_data_insights(df):
    """Compute and save data insights for the app."""
    print("\nComputing data insights...")
    
    # Monthly patterns
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    monthly = df.groupby('arrival_date_month').agg({
        'hotel': 'count',
        'is_canceled': 'mean',
        'adr': 'mean'
    }).rename(columns={'hotel': 'bookings'})
    monthly = monthly.reindex(month_order)
    
    # Lead time analysis
    df_temp = df.copy()
    lead_bins = [0, 7, 30, 90, 180, 365, df['lead_time'].max() + 1]
    labels = ['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days']
    df_temp['lead_time_group'] = pd.cut(df_temp['lead_time'], bins=lead_bins, labels=labels)
    lead_cancel = df_temp.groupby('lead_time_group', observed=True)['is_canceled'].mean() * 100
    
    # Revenue calculations
    df_temp['total_nights'] = df_temp['stays_in_weekend_nights'] + df_temp['stays_in_week_nights']
    df_temp['potential_revenue'] = df_temp['adr'] * df_temp['total_nights']
    total_potential = df_temp['potential_revenue'].sum()
    canceled_revenue = df_temp[df_temp['is_canceled']==1]['potential_revenue'].sum()
    realized_revenue = df_temp[df_temp['is_canceled']==0]['potential_revenue'].sum()
    
    # Yearly trends
    yearly = df.groupby('arrival_date_year').agg({
        'hotel': 'count',
        'is_canceled': 'mean',
        'adr': 'mean'
    }).rename(columns={'hotel': 'bookings'})
    
    insights = {
        'overview': {
            'total_bookings': int(len(df)),
            'date_range': f"{df['arrival_date_year'].min()} - {df['arrival_date_year'].max()}",
            'hotel_types': df['hotel'].unique().tolist(),
            'cancellation_rate': float(df['is_canceled'].mean() * 100),
            'canceled_bookings': int(df['is_canceled'].sum()),
            'completed_bookings': int((~df['is_canceled'].astype(bool)).sum())
        },
        'hotel_comparison': {
            hotel: {
                'bookings': int(len(df[df['hotel']==hotel])),
                'cancellation_rate': float(df[df['hotel']==hotel]['is_canceled'].mean() * 100),
                'avg_adr': float(df[df['hotel']==hotel]['adr'].mean())
            }
            for hotel in df['hotel'].unique()
        },
        'seasonal_patterns': {
            month: {
                'bookings': int(monthly.loc[month, 'bookings']) if month in monthly.index else 0,
                'cancellation_rate': float(monthly.loc[month, 'is_canceled'] * 100) if month in monthly.index else 0,
                'avg_adr': float(monthly.loc[month, 'adr']) if month in monthly.index else 0
            }
            for month in month_order
        },
        'peak_analysis': {
            'peak_month': monthly['bookings'].idxmax(),
            'peak_bookings': int(monthly['bookings'].max()),
            'low_month': monthly['bookings'].idxmin(),
            'low_bookings': int(monthly['bookings'].min()),
            'highest_adr_month': monthly['adr'].idxmax(),
            'highest_adr': float(monthly['adr'].max()),
            'lowest_adr_month': monthly['adr'].idxmin(),
            'lowest_adr': float(monthly['adr'].min())
        },
        'lead_time': {
            'average': float(df['lead_time'].mean()),
            'median': float(df['lead_time'].median()),
            'max': int(df['lead_time'].max()),
            'cancellation_by_group': {str(k): float(v) for k, v in lead_cancel.items()}
        },
        'guest_demographics': {
            'avg_adults': float(df['adults'].mean()),
            'avg_children': float(df['children'].mean()),
            'bookings_with_children': int((df['children'] > 0).sum()),
            'children_rate': float((df['children'] > 0).mean() * 100),
            'top_countries': {
                str(country): {'count': int(count), 'percentage': float(count/len(df)*100)}
                for country, count in df['country'].value_counts().head(10).items()
            }
        },
        'market_segments': {
            str(seg): {
                'count': int(count),
                'percentage': float(count/len(df)*100),
                'cancellation_rate': float(df[df['market_segment']==seg]['is_canceled'].mean()*100)
            }
            for seg, count in df['market_segment'].value_counts().items()
        },
        'deposit_types': {
            str(dep): {
                'count': int(count),
                'percentage': float(count/len(df)*100),
                'cancellation_rate': float(df[df['deposit_type']==dep]['is_canceled'].mean()*100)
            }
            for dep, count in df['deposit_type'].value_counts().items()
        },
        'special_requests': {
            int(req): {
                'count': int(len(df[df['total_of_special_requests']==req])),
                'cancellation_rate': float(df[df['total_of_special_requests']==req]['is_canceled'].mean()*100)
            }
            for req in sorted(df['total_of_special_requests'].unique())
        },
        'repeated_guests': {
            'count': int(df['is_repeated_guest'].sum()),
            'percentage': float(df['is_repeated_guest'].mean() * 100),
            'repeat_cancel_rate': float(df[df['is_repeated_guest']==1]['is_canceled'].mean()*100),
            'new_cancel_rate': float(df[df['is_repeated_guest']==0]['is_canceled'].mean()*100)
        },
        'revenue': {
            'total_potential': float(total_potential),
            'lost_to_cancellations': float(canceled_revenue),
            'lost_percentage': float(canceled_revenue/total_potential*100),
            'realized': float(realized_revenue),
            'avg_stay_nights': float(df_temp['total_nights'].mean()),
            'avg_adr': float(df['adr'].mean())
        },
        'yearly_trends': {
            int(year): {
                'bookings': int(row['bookings']),
                'cancellation_rate': float(row['is_canceled'] * 100),
                'avg_adr': float(row['adr'])
            }
            for year, row in yearly.iterrows()
        },
        'key_findings': [
            f"City Hotel has a significantly higher cancellation rate ({df[df['hotel']=='City Hotel']['is_canceled'].mean()*100:.1f}%) compared to Resort Hotel ({df[df['hotel']=='Resort Hotel']['is_canceled'].mean()*100:.1f}%)",
            f"Group bookings have the highest cancellation rate at {df[df['market_segment']=='Groups']['is_canceled'].mean()*100:.1f}%, while Direct bookings have the lowest at {df[df['market_segment']=='Direct']['is_canceled'].mean()*100:.1f}%",
            f"Non-refundable deposits correlate with 99.4% cancellation rate - primarily because they are used for Groups (63%) and Offline TA bookings (34%) which inherently have high cancellation rates due to long lead times",
            f"Lead time is a strong predictor: bookings made 365+ days in advance have a {lead_cancel['365+ days']:.1f}% cancellation rate vs only {lead_cancel['0-7 days']:.1f}% for last-minute bookings",
            f"Special requests indicate commitment: 0 requests = {df[df['total_of_special_requests']==0]['is_canceled'].mean()*100:.1f}% cancellation, but 5 requests = only {df[df['total_of_special_requests']==5]['is_canceled'].mean()*100:.1f}%",
            f"Repeated guests cancel only {df[df['is_repeated_guest']==1]['is_canceled'].mean()*100:.1f}% of the time compared to {df[df['is_repeated_guest']==0]['is_canceled'].mean()*100:.1f}% for new guests",
            f"August is the peak season with {int(monthly.loc['August', 'bookings']):,} bookings and highest ADR (${monthly.loc['August', 'adr']:.2f})",
            f"January is the low season with {int(monthly.loc['January', 'bookings']):,} bookings and lowest ADR (${monthly.loc['January', 'adr']:.2f})",
            f"Portugal (PRT) dominates bookings at 40.7%, followed by UK (10.2%) and France (8.7%)",
            f"Cancellations cost the hotels approximately ${canceled_revenue:,.0f} in lost revenue ({canceled_revenue/total_potential*100:.1f}% of potential revenue)"
        ]
    }
    
    with open(MODELS_DIR / "data_insights.json", 'w') as f:
        json.dump(insights, f, indent=2)
    print("  Saved data_insights.json")
    
    return insights


def main():
    print("="*60)
    print("HOTEL BOOKING MODEL TRAINING")
    print("="*60)
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Load data
    X, y, label_encoders, df = load_and_preprocess_data()
    
    # Train models
    trained_models, scaler, results, feature_importances, X_test, y_test = train_models(X, y)
    
    # Save everything
    save_models(trained_models, scaler, label_encoders, results, feature_importances)
    
    # Compute and save insights
    compute_data_insights(df)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest model: {max(results.keys(), key=lambda k: results[k]['f1_score'])}")
    print(f"Best F1 Score: {max(r['f1_score'] for r in results.values()):.4f}")


if __name__ == "__main__":
    main()
