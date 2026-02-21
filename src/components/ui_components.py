"""
Reusable UI components for the Streamlit application.
Provides consistent styling and layout components.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import COLORS, ICONS, LEARNING_OBJECTIVES


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
        
        /* Global Styles */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom header styling */
        .main-header {
            background: linear-gradient(135deg, #1E3A5F 0%, #3D5A80 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
        }
        
        .main-header h1 {
            margin: 0;
            font-weight: 700;
            font-size: 2.2rem;
        }
        
        .main-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        /* Section headers */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 1rem 0;
            border-bottom: 2px solid #1E3A5F;
            margin-bottom: 1.5rem;
        }
        
        .section-header h2 {
            margin: 0;
            color: #1E3A5F;
            font-weight: 600;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #1E3A5F;
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1E3A5F;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #7F8C8D;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-delta {
            font-size: 0.85rem;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            display: inline-block;
        }
        
        .metric-delta.positive {
            background: #E8F8F0;
            color: #2ECC71;
        }
        
        .metric-delta.negative {
            background: #FDEDEC;
            color: #E74C3C;
        }
        
        /* Info boxes */
        .info-box {
            background: #E3F2FD;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin: 1rem 0;
            border-left: 4px solid #1E3A5F;
        }
        
        .info-box.success {
            background: #E8F8F0;
            border-left-color: #2ECC71;
        }
        
        .info-box.warning {
            background: #FFF8E1;
            border-left-color: #F39C12;
        }
        
        .info-box.error {
            background: #FDEDEC;
            border-left-color: #E74C3C;
        }
        
        .info-box h4 {
            margin: 0 0 0.5rem 0;
            color: #1E3A5F;
            font-weight: 600;
        }
        
        .info-box p {
            margin: 0;
            color: #2C3E50;
        }
        
        /* Learning objectives */
        .learning-objectives {
            background: linear-gradient(135deg, #F8F9FA 0%, #E3F2FD 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .learning-objectives h4 {
            color: #1E3A5F;
            margin: 0 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .learning-objectives ul {
            margin: 0;
            padding-left: 1.5rem;
        }
        
        .learning-objectives li {
            margin: 0.5rem 0;
            color: #2C3E50;
        }
        
        /* Code explanation boxes */
        .code-explanation {
            background: #2C3E50;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin: 1rem 0;
            color: #ECF0F1;
            font-family: 'Fira Code', monospace;
        }
        
        /* Progress indicator */
        .progress-step {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem 0;
        }
        
        .progress-step .step-number {
            background: #1E3A5F;
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
        }
        
        .progress-step.completed .step-number {
            background: #2ECC71;
        }
        
        .progress-step.active .step-number {
            background: #EE6C4D;
        }
        
        /* Sidebar styling */
        .sidebar-section {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
        
        /* Table styling */
        .dataframe {
            font-size: 0.9rem !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #1E3A5F 0%, #3D5A80 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: white;
            border-radius: 8px 8px 0 0;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: #1E3A5F;
            color: white;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: #F8F9FA;
            border-radius: 8px;
            font-weight: 500;
        }
        
        /* Material Icons class */
        .material-symbols-outlined {
            font-family: 'Material Symbols Outlined';
            font-weight: normal;
            font-style: normal;
            font-size: 24px;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            display: inline-block;
            white-space: nowrap;
            word-wrap: normal;
            direction: ltr;
            vertical-align: middle;
        }
        
        /* Enhanced paragraph styling */
        .styled-text {
            background: linear-gradient(135deg, #F8FAFC 0%, #EDF2F7 100%);
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #3D5A80;
            line-height: 1.7;
            color: #2D3748;
            font-size: 1rem;
        }
        
        /* Insight cards */
        .insight-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-top: 3px solid #1E3A5F;
        }
        
        .insight-card .insight-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            color: #1E3A5F;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .insight-card .insight-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A5F;
            margin: 0.5rem 0;
        }
        
        .insight-card .insight-description {
            color: #64748B;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* Key finding boxes */
        .finding-box {
            background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            color: white;
        }
        
        .finding-box .finding-icon {
            font-size: 2rem;
            margin-bottom: 0.75rem;
        }
        
        .finding-box .finding-title {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        
        .finding-box .finding-detail {
            opacity: 0.9;
            line-height: 1.6;
        }
        
        /* Stats grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .stat-item {
            background: white;
            border-radius: 10px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-bottom: 3px solid #2C7A7B;
        }
        
        .stat-item .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1E3A5F;
        }
        
        .stat-item .stat-label {
            color: #64748B;
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }
        
        /* Comparison cards */
        .comparison-row {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .comparison-card {
            flex: 1;
            background: white;
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        
        .comparison-card.highlight {
            border: 2px solid #EE6C4D;
        }
        
        .comparison-card .comp-label {
            font-size: 0.85rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .comparison-card .comp-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1E3A5F;
            margin: 0.5rem 0;
        }
        
        /* Trend indicator */
        .trend-up { color: #38A169; }
        .trend-down { color: #E53E3E; }
        .trend-neutral { color: #718096; }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = ""):
    """Render the main page header."""
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, icon: str = "analytics"):
    """Render a section header with icon."""
    st.markdown(f"""
    <div class="section-header">
        <span class="material-symbols-outlined">{icon}</span>
        <h2>{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: bool = True
):
    """Render a metric card."""
    delta_html = ""
    if delta:
        delta_class = "positive" if delta_positive else "negative"
        delta_html = f'<span class="metric-delta {delta_class}">{delta}</span>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_info_box(
    content: str,
    title: Optional[str] = None,
    box_type: str = "info"
):
    """
    Render an information box.
    
    Parameters:
    -----------
    content : str
        The main content text.
    title : str, optional
        Box title.
    box_type : str
        Type of box: 'info', 'success', 'warning', 'error'.
    """
    title_html = f"<h4>{title}</h4>" if title else ""
    st.markdown(f"""
    <div class="info-box {box_type}">
        {title_html}
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def render_learning_objectives(section: str):
    """Render learning objectives for a section."""
    if section in LEARNING_OBJECTIVES:
        objectives = LEARNING_OBJECTIVES[section]
        objectives_html = "".join([f"<li>{obj}</li>" for obj in objectives])
        st.markdown(f"""
        <div class="learning-objectives">
            <h4>
                <span class="material-symbols-outlined">school</span>
                Learning Objectives
            </h4>
            <ul>
                {objectives_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)


def render_methodology_explanation(
    title: str,
    description: str,
    steps: List[str],
    formula: Optional[str] = None
):
    """
    Render a methodology explanation box.
    
    Parameters:
    -----------
    title : str
        Method name.
    description : str
        Brief description.
    steps : List[str]
        Steps involved.
    formula : str, optional
        Mathematical formula if applicable.
    """
    steps_html = "".join([
        f'<div class="progress-step"><span class="step-number">{i+1}</span>{step}</div>'
        for i, step in enumerate(steps)
    ])
    
    formula_html = f'<div class="code-explanation">{formula}</div>' if formula else ""
    
    with st.expander(f"Methodology: {title}", expanded=False):
        st.markdown(f"""
        <div style="padding: 1rem;">
            <p>{description}</p>
            <h4>Steps:</h4>
            {steps_html}
            {formula_html}
        </div>
        """, unsafe_allow_html=True)


def render_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Render sidebar filter controls.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to create filters for.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary of filter selections.
    """
    st.sidebar.markdown("### Filters")
    
    filters = {}
    
    # Hotel type filter
    hotels = df['hotel'].unique().tolist()
    filters['hotel'] = st.sidebar.multiselect(
        "Hotel Type",
        options=hotels,
        default=hotels
    )
    
    # Year filter
    years = sorted(df['arrival_date_year'].unique().tolist())
    filters['year'] = st.sidebar.multiselect(
        "Year",
        options=years,
        default=years
    )
    
    # Month filter
    months = df['arrival_date_month'].cat.categories.tolist()
    filters['month'] = st.sidebar.multiselect(
        "Month",
        options=months,
        default=months
    )
    
    # Cancellation filter
    filters['cancellation'] = st.sidebar.radio(
        "Booking Status",
        options=['All', 'Completed', 'Canceled'],
        horizontal=True
    )
    
    return filters


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply sidebar filters to dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe.
    filters : Dict[str, Any]
        Filter selections from sidebar.
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe.
    """
    filtered = df.copy()
    
    if filters.get('hotel'):
        filtered = filtered[filtered['hotel'].isin(filters['hotel'])]
    
    if filters.get('year'):
        filtered = filtered[filtered['arrival_date_year'].isin(filters['year'])]
    
    if filters.get('month'):
        filtered = filtered[filtered['arrival_date_month'].isin(filters['month'])]
    
    if filters.get('cancellation') == 'Completed':
        filtered = filtered[filtered['is_canceled'] == 0]
    elif filters.get('cancellation') == 'Canceled':
        filtered = filtered[filtered['is_canceled'] == 1]
    
    return filtered


def render_dataframe_preview(
    df: pd.DataFrame,
    title: str = "Data Preview",
    rows: int = 10
):
    """Render a styled dataframe preview."""
    render_section_header(title, "table_view")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{len(df.columns)}")
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    st.dataframe(df.head(rows), use_container_width=True)


def render_code_block(code: str, language: str = "python"):
    """Render a code block with syntax highlighting."""
    st.code(code, language=language)


def create_download_button(
    df: pd.DataFrame,
    filename: str = "data.csv",
    label: str = "Download Data"
):
    """Create a download button for dataframe."""
    csv = df.to_csv(index=False)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def render_tabs(tab_names: List[str]) -> List:
    """Create styled tabs and return tab objects."""
    return st.tabs(tab_names)


def render_kpi_row(metrics: List[Dict[str, Any]]):
    """
    Render a row of KPI metrics.
    
    Parameters:
    -----------
    metrics : List[Dict]
        List of dictionaries with 'label', 'value', and optional 'delta'.
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            delta = metric.get('delta')
            delta_positive = metric.get('delta_positive', True)
            render_metric_card(
                metric['label'],
                metric['value'],
                delta,
                delta_positive
            )


def render_progress_stepper(
    steps: List[str],
    current_step: int
):
    """
    Render a progress stepper component.
    
    Parameters:
    -----------
    steps : List[str]
        List of step names.
    current_step : int
        Current active step (0-indexed).
    """
    html_steps = ""
    for i, step in enumerate(steps):
        if i < current_step:
            status = "completed"
        elif i == current_step:
            status = "active"
        else:
            status = ""
        
        html_steps += f"""
        <div class="progress-step {status}">
            <span class="step-number">{i + 1}</span>
            <span>{step}</span>
        </div>
        """
    
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        {html_steps}
    </div>
    """, unsafe_allow_html=True)


def render_styled_text(text: str, icon: Optional[str] = None):
    """
    Render styled paragraph text with better UI.
    
    Parameters:
    -----------
    text : str
        The text content to display.
    icon : str, optional
        Material icon name to display.
    """
    icon_html = f'<span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 8px; color: #3D5A80;">{icon}</span>' if icon else ''
    st.markdown(f"""
    <div class="styled-text">
        {icon_html}{text}
    </div>
    """, unsafe_allow_html=True)


def render_insight_card(
    title: str,
    value: str,
    description: str,
    icon: str = "insights"
):
    """
    Render an insight card with value highlight.
    
    Parameters:
    -----------
    title : str
        Insight title.
    value : str
        Main value to highlight.
    description : str
        Description text.
    icon : str
        Material icon name.
    """
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">
            <span class="material-symbols-outlined">{icon}</span>
            {title}
        </div>
        <div class="insight-value">{value}</div>
        <div class="insight-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def render_finding_box(
    title: str,
    detail: str,
    icon: str = "lightbulb"
):
    """
    Render a key finding box.
    
    Parameters:
    -----------
    title : str
        Finding title/headline.
    detail : str
        Detailed description.
    icon : str
        Material icon name.
    """
    st.markdown(f"""
    <div class="finding-box">
        <div class="finding-icon">
            <span class="material-symbols-outlined">{icon}</span>
        </div>
        <div class="finding-title">{title}</div>
        <div class="finding-detail">{detail}</div>
    </div>
    """, unsafe_allow_html=True)


def render_stats_grid(stats: List[Dict[str, str]]):
    """
    Render a grid of statistics.
    
    Parameters:
    -----------
    stats : List[Dict]
        List of dicts with 'value' and 'label' keys.
    """
    items_html = "".join([
        f'<div class="stat-item"><div class="stat-value">{s["value"]}</div><div class="stat-label">{s["label"]}</div></div>'
        for s in stats
    ])
    st.markdown(f"""
    <div class="stats-grid">
        {items_html}
    </div>
    """, unsafe_allow_html=True)


def render_comparison(
    label1: str,
    value1: str,
    label2: str,
    value2: str,
    highlight_first: bool = False
):
    """
    Render a side-by-side comparison.
    
    Parameters:
    -----------
    label1, label2 : str
        Labels for each value.
    value1, value2 : str
        Values to compare.
    highlight_first : bool
        Whether to highlight the first card.
    """
    highlight_class = "highlight" if highlight_first else ""
    st.markdown(f"""
    <div class="comparison-row">
        <div class="comparison-card {highlight_class}">
            <div class="comp-label">{label1}</div>
            <div class="comp-value">{value1}</div>
        </div>
        <div class="comparison-card {'highlight' if not highlight_first else ''}">
            <div class="comp-label">{label2}</div>
            <div class="comp-value">{value2}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
