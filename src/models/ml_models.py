"""
Machine Learning models for hotel booking analysis.
Includes preprocessing, training, evaluation, and prediction functionality.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RANDOM_STATE, TEST_SIZE, CV_FOLDS, FEATURES_FOR_CANCELLATION


class HotelBookingPreprocessor:
    """
    Preprocessor for hotel booking data for ML models.
    
    This class handles:
    - Feature selection
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.categorical_columns = [
            'hotel', 'meal', 'market_segment', 'distribution_channel',
            'reserved_room_type', 'deposit_type', 'customer_type'
        ]
        self.numerical_features = FEATURES_FOR_CANCELLATION
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: str = 'is_canceled'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe.
        target_column : str
            Name of target column.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            X (features) and y (target) arrays.
        """
        df_processed = df.copy()
        
        # Handle missing values
        df_processed['children'] = df_processed['children'].fillna(0)
        df_processed['agent'] = df_processed['agent'].fillna(0)
        df_processed['company'] = df_processed['company'].fillna(0)
        
        # Encode categorical features
        encoded_features = []
        for col in self.categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[f'{col}_encoded'] = le.fit_transform(
                    df_processed[col].astype(str)
                )
                self.label_encoders[col] = le
                encoded_features.append(f'{col}_encoded')
        
        # Combine numerical and encoded categorical features
        self.feature_names = self.numerical_features + encoded_features
        
        # Extract features and target
        X = df_processed[self.feature_names].values
        y = df_processed[target_column].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe.
        
        Returns:
        --------
        np.ndarray
            Transformed features.
        """
        df_processed = df.copy()
        
        # Handle missing values
        df_processed['children'] = df_processed['children'].fillna(0)
        df_processed['agent'] = df_processed['agent'].fillna(0)
        df_processed['company'] = df_processed['company'].fillna(0)
        
        # Encode categorical features
        for col in self.categorical_columns:
            if col in df_processed.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                df_processed[f'{col}_encoded'] = df_processed[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Extract features
        X = df_processed[self.feature_names].values
        
        # Scale features
        X = self.scaler.transform(X)
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names


class CancellationPredictionModel:
    """
    Model for predicting hotel booking cancellations.
    
    Supports multiple algorithms:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Decision Tree
    - K-Nearest Neighbors
    """
    
    AVAILABLE_MODELS = {
        'logistic_regression': {
            'name': 'Logistic Regression',
            'class': LogisticRegression,
            'params': {'random_state': RANDOM_STATE, 'max_iter': 1000}
        },
        'random_forest': {
            'name': 'Random Forest',
            'class': RandomForestClassifier,
            'params': {'random_state': RANDOM_STATE, 'n_estimators': 100, 'n_jobs': -1}
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'class': GradientBoostingClassifier,
            'params': {'random_state': RANDOM_STATE, 'n_estimators': 100}
        },
        'decision_tree': {
            'name': 'Decision Tree',
            'class': DecisionTreeClassifier,
            'params': {'random_state': RANDOM_STATE, 'max_depth': 10}
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'class': KNeighborsClassifier,
            'params': {'n_neighbors': 5, 'n_jobs': -1}
        }
    }
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use.
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        model_config = self.AVAILABLE_MODELS[model_type]
        self.model_name = model_config['name']
        self.model = model_config['class'](**model_config['params'])
        self.preprocessor = HotelBookingPreprocessor()
        self.is_fitted = False
        self.metrics = {}
        
    def train(
        self,
        df: pd.DataFrame,
        target_column: str = 'is_canceled',
        test_size: float = TEST_SIZE
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data.
        target_column : str
            Name of target column.
        test_size : float
            Proportion of data to use for testing.
        
        Returns:
        --------
        Dict[str, Any]
            Training results and metrics.
        """
        # Preprocess data
        X, y = self.preprocessor.fit_transform(df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        # Store data for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        self.metrics = self.evaluate(X_test, y_test)
        
        # Add cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=CV_FOLDS, scoring='accuracy'
        )
        self.metrics['cv_accuracy_mean'] = cv_scores.mean()
        self.metrics['cv_accuracy_std'] = cv_scores.std()
        
        return self.metrics
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True labels.
        
        Returns:
        --------
        Dict[str, Any]
            Evaluation metrics.
        """
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_prob)
            metrics['fpr'], metrics['tpr'], _ = roc_curve(y, y_prob)
            metrics['y_prob'] = y_prob
        
        metrics['y_pred'] = y_pred
        metrics['y_true'] = y
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Predictions and probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet.")
        
        X = self.preprocessor.transform(df)
        predictions = self.model.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            probabilities = predictions.astype(float)
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
        --------
        Dict[str, float] or None
            Feature importance dictionary.
        """
        if not self.is_fitted:
            return None
        
        feature_names = self.preprocessor.get_feature_names()
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return None
        
        return dict(zip(feature_names, importances))
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if not self.is_fitted:
            return "Model not trained"
        
        return classification_report(
            self.y_test,
            self.model.predict(self.X_test),
            target_names=['Not Canceled', 'Canceled']
        )


class ModelComparison:
    """
    Compare multiple models and select the best one.
    """
    
    def __init__(self, model_types: List[str] = None):
        """
        Initialize model comparison.
        
        Parameters:
        -----------
        model_types : List[str], optional
            List of model types to compare.
        """
        if model_types is None:
            model_types = list(CancellationPredictionModel.AVAILABLE_MODELS.keys())
        
        self.model_types = model_types
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_type = None
        
    def run_comparison(
        self,
        df: pd.DataFrame,
        target_column: str = 'is_canceled'
    ) -> pd.DataFrame:
        """
        Train and compare all models.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data.
        target_column : str
            Name of target column.
        
        Returns:
        --------
        pd.DataFrame
            Comparison results.
        """
        for model_type in self.model_types:
            model = CancellationPredictionModel(model_type)
            metrics = model.train(df, target_column)
            
            self.models[model_type] = model
            self.results[model_type] = {
                'Model': model.model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics.get('roc_auc', 'N/A'),
                'CV Accuracy (mean)': metrics['cv_accuracy_mean'],
                'CV Accuracy (std)': metrics['cv_accuracy_std']
            }
        
        # Find best model based on F1 score
        best_f1 = 0
        for model_type, result in self.results.items():
            if result['F1 Score'] > best_f1:
                best_f1 = result['F1 Score']
                self.best_model_type = model_type
                self.best_model = self.models[model_type]
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results.values())
        results_df = results_df.sort_values('F1 Score', ascending=False)
        
        return results_df
    
    def get_best_model(self) -> CancellationPredictionModel:
        """Get the best performing model."""
        return self.best_model
    
    def get_model(self, model_type: str) -> CancellationPredictionModel:
        """Get a specific model by type."""
        return self.models.get(model_type)


def calculate_business_metrics(
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    avg_booking_value: float = 100.0
) -> Dict[str, float]:
    """
    Calculate business-relevant metrics from predictions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original data.
    predictions : np.ndarray
        Model predictions.
    probabilities : np.ndarray
        Prediction probabilities.
    avg_booking_value : float
        Average value per booking.
    
    Returns:
    --------
    Dict[str, float]
        Business metrics.
    """
    true_cancellations = df['is_canceled'].values
    
    # True positives - correctly predicted cancellations
    tp = np.sum((predictions == 1) & (true_cancellations == 1))
    
    # False positives - incorrectly predicted as cancellations
    fp = np.sum((predictions == 1) & (true_cancellations == 0))
    
    # False negatives - missed cancellations
    fn = np.sum((predictions == 0) & (true_cancellations == 1))
    
    # Potential revenue saved by predicting cancellations
    potential_savings = tp * avg_booking_value
    
    # Revenue potentially lost due to overbooking (false positives)
    potential_loss = fp * avg_booking_value * 0.5  # Assume 50% loss for overbooking
    
    # Missed opportunity cost
    missed_opportunity = fn * avg_booking_value
    
    return {
        'predicted_cancellations': int(np.sum(predictions)),
        'actual_cancellations': int(np.sum(true_cancellations)),
        'correctly_predicted': int(tp),
        'false_alarms': int(fp),
        'missed_cancellations': int(fn),
        'potential_savings': potential_savings,
        'potential_overbooking_loss': potential_loss,
        'net_benefit': potential_savings - potential_loss,
        'avg_cancellation_probability': np.mean(probabilities)
    }


def get_model_explanation(model_type: str) -> Dict[str, str]:
    """
    Get educational explanation for a model type.
    
    Parameters:
    -----------
    model_type : str
        Type of model.
    
    Returns:
    --------
    Dict[str, str]
        Model explanation including description, pros, cons, and use cases.
    """
    explanations = {
        'logistic_regression': {
            'description': """
            Logistic Regression is a statistical model that uses a logistic function to model 
            a binary dependent variable. It estimates the probability of an event occurring 
            based on a linear combination of input features.
            """,
            'pros': [
                'Simple and interpretable',
                'Fast to train and predict',
                'Provides probability estimates',
                'Works well with linearly separable data',
                'Less prone to overfitting'
            ],
            'cons': [
                'Assumes linear relationship',
                'May underperform on complex patterns',
                'Sensitive to outliers',
                'Requires feature scaling'
            ],
            'use_case': 'Best when you need interpretable results and baseline performance.'
        },
        'random_forest': {
            'description': """
            Random Forest is an ensemble learning method that constructs multiple decision trees 
            during training and outputs the mode of their predictions. It reduces overfitting by 
            averaging multiple trees built on random subsets of data and features.
            """,
            'pros': [
                'Handles non-linear relationships',
                'Robust to outliers and noise',
                'Provides feature importance',
                'Less prone to overfitting than single trees',
                'Works well with high-dimensional data'
            ],
            'cons': [
                'Can be slow for large datasets',
                'Less interpretable than single trees',
                'May require more memory',
                'Can overfit on noisy data'
            ],
            'use_case': 'Best for complex datasets where accuracy is important and some interpretability is still needed.'
        },
        'gradient_boosting': {
            'description': """
            Gradient Boosting builds models sequentially, with each new model correcting errors 
            made by the previous ones. It combines multiple weak learners (typically shallow trees) 
            to create a strong predictive model.
            """,
            'pros': [
                'Often achieves best accuracy',
                'Handles various data types',
                'Provides feature importance',
                'Flexible loss functions',
                'Good for structured/tabular data'
            ],
            'cons': [
                'Slower to train than Random Forest',
                'Requires careful hyperparameter tuning',
                'Can overfit if not regularized',
                'Less interpretable'
            ],
            'use_case': 'Best when maximum predictive accuracy is required and training time is not a constraint.'
        },
        'decision_tree': {
            'description': """
            Decision Tree creates a model that predicts the target value by learning simple 
            decision rules inferred from the features. It splits the data based on feature 
            values to create a tree-like structure.
            """,
            'pros': [
                'Highly interpretable',
                'Handles non-linear relationships',
                'No feature scaling required',
                'Fast to train and predict',
                'Visual representation possible'
            ],
            'cons': [
                'Prone to overfitting',
                'Unstable (small changes cause different trees)',
                'May create biased trees',
                'Often less accurate than ensembles'
            ],
            'use_case': 'Best for exploratory analysis and when model explainability is critical.'
        },
        'knn': {
            'description': """
            K-Nearest Neighbors classifies samples based on the majority class among their K 
            nearest neighbors in the feature space. It's a non-parametric, lazy learning algorithm 
            that stores all training data and makes predictions at inference time.
            """,
            'pros': [
                'Simple to understand',
                'No training phase',
                'Naturally handles multi-class',
                'Adapts to any data distribution',
                'No assumptions about data'
            ],
            'cons': [
                'Slow prediction for large data',
                'Sensitive to irrelevant features',
                'Requires feature scaling',
                'Memory intensive',
                'Sensitive to the choice of K'
            ],
            'use_case': 'Best for smaller datasets and when decision boundaries are irregular.'
        }
    }
    
    return explanations.get(model_type, {
        'description': 'No explanation available.',
        'pros': [],
        'cons': [],
        'use_case': 'N/A'
    })
