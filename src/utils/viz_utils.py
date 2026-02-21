"""
Visualization utilities using Plotly for interactive charts.
All charts follow a consistent theme and styling.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import COLORS, PLOTLY_COLORS, SEQUENTIAL_COLORS, MONTH_ORDER


def get_plotly_template() -> Dict[str, Any]:
    """
    Get consistent Plotly template for all charts.
    
    Returns:
    --------
    Dict[str, Any]
        Plotly layout configuration.
    """
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {
            'family': 'Inter, sans-serif',
            'color': COLORS['text_primary'],
            'size': 12
        },
        'xaxis': {
            'gridcolor': '#E8E8E8',
            'linecolor': '#E8E8E8',
            'showgrid': True,
            'gridwidth': 1
        },
        'yaxis': {
            'gridcolor': '#E8E8E8',
            'linecolor': '#E8E8E8',
            'showgrid': True,
            'gridwidth': 1
        },
        'colorway': PLOTLY_COLORS,
        'hoverlabel': {
            'bgcolor': 'white',
            'font_size': 12,
            'font_family': 'Inter, sans-serif'
        },
        'margin': {'t': 60, 'l': 60, 'r': 30, 'b': 60}
    }


def apply_chart_style(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent styling to a plotly figure."""
    fig.update_layout(
        **get_plotly_template(),
        title={
            'text': title,
            'font': {'size': 18, 'color': COLORS['primary']},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    return fig


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    bins: int = 30,
    show_stats: bool = True
) -> go.Figure:
    """
    Create a histogram with optional KDE overlay.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to plot.
    title : str, optional
        Chart title.
    bins : int
        Number of histogram bins.
    show_stats : bool
        Whether to show statistical annotations.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"Distribution of {column}"
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=df[column],
        nbinsx=bins,
        name='Count',
        marker_color=COLORS['primary'],
        opacity=0.7
    ))
    
    apply_chart_style(fig, title)
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Frequency',
        showlegend=False
    )
    
    if show_stats:
        mean_val = df[column].mean()
        median_val = df[column].median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS['accent'],
                      annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dot", line_color=COLORS['success'],
                      annotation_text=f"Median: {median_val:.2f}")
    
    return fig


def plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    top_n: int = 10,
    horizontal: bool = True,
    show_percentage: bool = True
) -> go.Figure:
    """
    Create a bar chart for categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to plot.
    title : str, optional
        Chart title.
    top_n : int
        Number of top categories to show.
    horizontal : bool
        Whether to use horizontal bars.
    show_percentage : bool
        Whether to show percentage labels.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"Distribution of {column}"
    
    value_counts = df[column].value_counts().head(top_n)
    percentages = (value_counts / len(df) * 100).round(1)
    
    if horizontal:
        fig = go.Figure(go.Bar(
            y=value_counts.index.astype(str),
            x=value_counts.values,
            orientation='h',
            marker_color=COLORS['primary'],
            text=[f"{p}%" for p in percentages] if show_percentage else None,
            textposition='auto'
        ))
        fig.update_layout(
            xaxis_title='Count',
            yaxis_title=column,
            yaxis={'categoryorder': 'total ascending'}
        )
    else:
        fig = go.Figure(go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            marker_color=COLORS['primary'],
            text=[f"{p}%" for p in percentages] if show_percentage else None,
            textposition='auto'
        ))
        fig.update_layout(
            xaxis_title=column,
            yaxis_title='Count'
        )
    
    apply_chart_style(fig, title)
    return fig


def plot_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    title: Optional[str] = None,
    color_by: Optional[str] = None,
    show_trend: bool = False
) -> go.Figure:
    """
    Create a time series line chart.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    date_column : str
        Column containing dates.
    value_column : str
        Column with values to plot.
    title : str, optional
        Chart title.
    color_by : str, optional
        Column to color lines by.
    show_trend : bool
        Whether to show trend line.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{value_column} Over Time"
    
    if color_by:
        fig = px.line(
            df, x=date_column, y=value_column, color=color_by,
            color_discrete_sequence=PLOTLY_COLORS
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[date_column],
            y=df[value_column],
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=6),
            name=value_column
        ))
        
        if show_trend:
            # Simple moving average trend
            window = min(7, len(df) // 4)
            if window > 1:
                trend = df[value_column].rolling(window=window, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=df[date_column],
                    y=trend,
                    mode='lines',
                    line=dict(color=COLORS['accent'], width=2, dash='dash'),
                    name=f'Trend (MA{window})'
                ))
    
    apply_chart_style(fig, title)
    return fig


def plot_monthly_pattern(
    df: pd.DataFrame,
    value_column: str = 'is_canceled',
    agg_func: str = 'count',
    title: Optional[str] = None,
    by_year: bool = False
) -> go.Figure:
    """
    Create a monthly pattern chart showing seasonality.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    value_column : str
        Column to aggregate.
    agg_func : str
        Aggregation function.
    title : str, optional
        Chart title.
    by_year : bool
        Whether to separate by year.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"Monthly {agg_func.title()} of Bookings"
    
    if by_year:
        grouped = df.groupby(['arrival_date_month', 'arrival_date_year']).agg(
            {value_column: agg_func}
        ).reset_index()
        
        fig = px.line(
            grouped,
            x='arrival_date_month',
            y=value_column,
            color='arrival_date_year',
            markers=True,
            color_discrete_sequence=PLOTLY_COLORS
        )
        fig.update_xaxes(categoryorder='array', categoryarray=MONTH_ORDER)
    else:
        grouped = df.groupby('arrival_date_month').agg(
            {value_column: agg_func}
        ).reindex(MONTH_ORDER)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grouped.index,
            y=grouped[value_column],
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor=f"rgba(30, 58, 95, 0.1)"
        ))
    
    apply_chart_style(fig, title)
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    agg_func: str = 'mean',
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a heatmap for two categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    x_column : str
        Column for x-axis.
    y_column : str
        Column for y-axis.
    value_column : str
        Column to aggregate for color.
    agg_func : str
        Aggregation function.
    title : str, optional
        Chart title.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{value_column} by {x_column} and {y_column}"
    
    pivot = df.pivot_table(
        values=value_column,
        index=y_column,
        columns=x_column,
        aggfunc=agg_func
    )
    
    # Order months if applicable
    if x_column == 'arrival_date_month':
        pivot = pivot.reindex(columns=[m for m in MONTH_ORDER if m in pivot.columns])
    if y_column == 'arrival_date_month':
        pivot = pivot.reindex([m for m in MONTH_ORDER if m in pivot.index])
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index.astype(str),
        colorscale=[[0, COLORS['background']], [0.5, COLORS['secondary']], [1, COLORS['primary']]],
        hoverongaps=False,
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    apply_chart_style(fig, title)
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "Feature Correlation Matrix"
) -> go.Figure:
    """
    Create a correlation heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    columns : List[str]
        Columns to include in correlation.
    title : str
        Chart title.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    corr_matrix = df[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9}
    ))
    
    apply_chart_style(fig, title)
    fig.update_layout(
        height=600,
        width=800
    )
    return fig


def plot_box_comparison(
    df: pd.DataFrame,
    numerical_column: str,
    categorical_column: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create box plots comparing numerical values across categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    numerical_column : str
        Numerical column for values.
    categorical_column : str
        Categorical column for grouping.
    title : str, optional
        Chart title.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"{numerical_column} by {categorical_column}"
    
    fig = px.box(
        df,
        x=categorical_column,
        y=numerical_column,
        color=categorical_column,
        color_discrete_sequence=PLOTLY_COLORS
    )
    
    apply_chart_style(fig, title)
    fig.update_layout(showlegend=False)
    
    # Order months if applicable
    if categorical_column == 'arrival_date_month':
        fig.update_xaxes(categoryorder='array', categoryarray=MONTH_ORDER)
    
    return fig


def plot_cancellation_analysis(
    df: pd.DataFrame,
    group_by: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create dual-axis chart showing count and cancellation rate.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    group_by : str
        Column to group by.
    title : str, optional
        Chart title.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"Bookings and Cancellation Rate by {group_by}"
    
    grouped = df.groupby(group_by).agg({
        'is_canceled': ['count', 'mean']
    }).reset_index()
    grouped.columns = [group_by, 'count', 'cancellation_rate']
    grouped['cancellation_rate'] = grouped['cancellation_rate'] * 100
    
    # Order months if applicable
    if group_by == 'arrival_date_month':
        grouped['month_order'] = grouped[group_by].map(
            {m: i for i, m in enumerate(MONTH_ORDER)}
        )
        grouped = grouped.sort_values('month_order')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=grouped[group_by].astype(str),
            y=grouped['count'],
            name='Total Bookings',
            marker_color=COLORS['primary'],
            opacity=0.7
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=grouped[group_by].astype(str),
            y=grouped['cancellation_rate'],
            name='Cancellation Rate (%)',
            line=dict(color=COLORS['accent'], width=3),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    apply_chart_style(fig, title)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Booking Count", secondary_y=False)
    fig.update_yaxes(title_text="Cancellation Rate (%)", secondary_y=True)
    
    return fig


def plot_pie_chart(
    df: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    hole: float = 0.4
) -> go.Figure:
    """
    Create a donut/pie chart.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to plot.
    title : str, optional
        Chart title.
    hole : float
        Size of hole for donut chart (0 for pie).
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    if title is None:
        title = f"Distribution of {column}"
    
    value_counts = df[column].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=value_counts.index.astype(str),
        values=value_counts.values,
        hole=hole,
        marker_colors=PLOTLY_COLORS[:len(value_counts)],
        textinfo='label+percent',
        textposition='outside'
    )])
    
    apply_chart_style(fig, title)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    title: str = "Feature Importance"
) -> go.Figure:
    """
    Create a horizontal bar chart for feature importance.
    
    Parameters:
    -----------
    feature_names : List[str]
        Names of features.
    importances : List[float]
        Importance values.
    title : str
        Chart title.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    # Sort by importance
    sorted_idx = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = [importances[i] for i in sorted_idx]
    
    # Color gradient based on importance
    colors = [COLORS['secondary'] if imp < np.mean(importances) 
              else COLORS['primary'] for imp in sorted_importance]
    
    fig = go.Figure(go.Bar(
        y=sorted_features,
        x=sorted_importance,
        orientation='h',
        marker_color=colors
    ))
    
    apply_chart_style(fig, title)
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=max(400, len(feature_names) * 25)
    )
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = ['Not Canceled', 'Canceled'],
    title: str = "Confusion Matrix"
) -> go.Figure:
    """
    Create a confusion matrix heatmap.
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix array.
    labels : List[str]
        Class labels.
    title : str
        Chart title.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    # Calculate percentages
    cm_pct = cm / cm.sum() * 100
    
    # Create text annotations
    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" 
             for j in range(len(labels))] for i in range(len(labels))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Predicted<br>{l}" for l in labels],
        y=[f"Actual<br>{l}" for l in labels],
        colorscale=[[0, '#E8F4F8'], [1, COLORS['primary']]],
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14},
        showscale=False
    ))
    
    apply_chart_style(fig, title)
    fig.update_layout(
        height=400,
        width=500
    )
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve"
) -> go.Figure:
    """
    Create ROC curve visualization.
    
    Parameters:
    -----------
    fpr : np.ndarray
        False positive rates.
    tpr : np.ndarray
        True positive rates.
    auc_score : float
        Area under curve score.
    title : str
        Chart title.
    
    Returns:
    --------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color=COLORS['primary'], width=3)
    ))
    
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    apply_chart_style(fig, title)
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.6, y=0.1),
        width=600,
        height=500
    )
    return fig
