from funcs import *
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from glob import glob as lsfiles
import numpy as np

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Influenza Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Initialize page in session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard'

# Initialize selected horizon
if 'selected_horizon' not in st.session_state:
    st.session_state.selected_horizon = 1
elif st.session_state.selected_horizon > 3:
    st.session_state.selected_horizon = 1

# ============================================================================
# THEME CONFIGURATION
# ============================================================================

THEMES = {
    'light': {
        'primary': '#2563eb',
        'secondary': '#64748b',
        'accent': '#0ea5e9',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'background': '#ffffff',
        'surface': '#f8fafc',
        'text': '#1e293b',
        'text_muted': '#64748b',
        'border': '#e2e8f0',
        'app_bg': '#fafafa',
        'sidebar_bg': 'white',
        'plotly_template': 'plotly_white',
        'plotly_bg': 'white',
        'plotly_grid': 'lightgray'
    },
    'dark': {
        'primary': '#3b82f6',
        'secondary': '#94a3b8',
        'accent': '#38bdf8',
        'success': '#34d399',
        'warning': '#fbbf24',
        'danger': '#f87171',
        'background': '#1e293b',
        'surface': '#334155',
        'text': '#f1f5f9',
        'text_muted': '#cbd5e1',
        'border': '#475569',
        'app_bg': '#0f172a',
        'sidebar_bg': '#1e293b',
        'plotly_template': 'plotly_dark',
        'plotly_bg': '#1e293b',
        'plotly_grid': '#334155'
    }
}

# Model color palette
MODEL_COLORS = {
    'MOBS-GLEAM_FLUH': '#2563eb',
    'NEU_ISI-AdaptiveEnsemble': '#dc2626',
    'NEU_ISI-FluBcast': '#16a34a'
}

# Get current theme colors
COLORS = THEMES[st.session_state.theme]

# ============================================================================
# THEME STYLES
# ============================================================================

def apply_theme_styles():
    """Apply CSS styles based on current theme"""
    colors = THEMES[st.session_state.theme]
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global styles */
        .stApp {{
            background-color: {colors['app_bg']};
        }}
        
        /* Typography */
        .stApp, .stMarkdown, p, span, div {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
            color: {colors['text']};
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 600;
            color: {colors['text']} !important;
            letter-spacing: -0.025em;
        }}
        
        /* Navigation buttons */
        .nav-button {{
            display: inline-block;
            padding: 0.5rem 1.5rem;
            background: {colors['surface']};
            border: 2px solid {colors['border']};
            border-radius: 6px;
            color: {colors['text']};
            text-decoration: none;
            font-weight: 500;
            margin: 0 0.25rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .nav-button:hover {{
            background: {colors['primary']};
            color: white;
            border-color: {colors['primary']};
        }}
        
        .nav-button.active {{
            background: {colors['primary']};
            color: white;
            border-color: {colors['primary']};
        }}
        
        /* Horizon selector buttons */
        .horizon-button {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: {colors['surface']};
            border: 2px solid {colors['border']};
            color: {colors['text']};
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .horizon-button:hover {{
            background: {colors['accent']};
            color: white;
            border-color: {colors['accent']};
        }}
        
        .horizon-button.selected {{
            background: {colors['primary']};
            color: white;
            border-color: {colors['primary']};
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {colors['sidebar_bg']};
            border-right: 1px solid {colors['border']};
        }}
        
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: {colors['text']};
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        /* Metrics */
        [data-testid="metric-container"] {{
            background: {colors['background']};
            border: 1px solid {colors['border']};
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        /* Custom containers */
        .stat-card {{
            background: {colors['background']};
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid {colors['border']};
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        .section-header {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {colors['text_muted']};
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        
        /* Model legend styling */
        .model-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
        }}
        
        /* Remove default Streamlit padding in some areas */
        .block-container {{
            padding-top: 1rem;
        }}
        
        /* Fix for navigation positioning - INCREASED PADDING */
        .main .block-container {{
            padding-top: 5rem;
        }}
        
        /* Additional padding for Streamlit Cloud deployment bar */
        .stApp > header {{
            top: 0;
        }}
        
        .main {{
            margin-top: 20px;
        }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA DEFINITIONS
# ============================================================================

states = [
    "United States",'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
    'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'West Virginia', 'Wisconsin', 'Wyoming'
]


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load and cache all data"""
    usa_gpd = gpd.read_file("states_21basic/states.shp")
    locations = pd.read_csv("locations.csv")
    
    models = ["MOBS-GLEAM_FLUH","NEU_ISI-AdaptiveEnsemble","NEU_ISI-FluBcast"]
    df_all_forecasts = []
    
    for model in models: 
        forecasts = sorted(lsfiles(f"forecasts/{model}/*"))
        
        df_forecasts = pd.concat([pd.read_csv(f) for f in forecasts])
        df_forecasts = df_forecasts[df_forecasts.output_type == "quantile"]
        df_forecasts['reference_date'] = pd.to_datetime(df_forecasts['reference_date'])
        df_forecasts['target_end_date'] = pd.to_datetime(df_forecasts['target_end_date'])
        df_forecasts['output_type_id'] = df_forecasts['output_type_id'].astype(float)
        # Remove date filter to get all available data
        df_forecasts = df_forecasts[df_forecasts.reference_date >= pd.to_datetime("2024-09-30")]
        df_forecasts['model'] = model
        
        # Calculate horizon (weeks ahead)
        #df_forecasts['horizon'] = ((df_forecasts['target_end_date'] - df_forecasts['reference_date']).dt.days / 7).round().astype(int)
        
        df_all_forecasts.append(df_forecasts.copy())
    
    df_all_forecasts = pd.concat(df_all_forecasts)
    
    df_target_data = pd.read_csv("target_surveillance/target-hospital-admissions.csv")
    df_target_data['date'] = pd.to_datetime(df_target_data['date'])
    # Remove date filter to get all available data
    df_target_data = df_target_data[df_target_data.date >= pd.to_datetime("2024-10-30")]


    #load evaluations data
    scores = ['WIS','WIS_ratio','MAPE']
    #load evaluations data
    models = ["MOBS-GLEAM_FLUH","NEU_ISI-AdaptiveEnsemble","NEU_ISI-FluBcast"]

    scores = ['WIS','WIS_ratio','MAPE']

    df_wis = pd.read_csv("evaluations/%s.csv" % "WIS")
    df_wis['score'] = "WIS"
    df_wis = df_wis.rename(columns = {"wis":"value"})

    df_mape = pd.read_csv("evaluations/%s.csv" % "MAPE")
    df_mape['score'] = "MAPE"
    df_mape = df_mape.rename(columns = {"MAPE":"value"})


    df_wis_ratio = pd.read_csv("evaluations/%s.csv" % "WIS_ratio")
    df_wis_ratio['score'] = "WIS_ratio"
    df_wis_ratio = df_wis_ratio.rename(columns = {"wis_ratio":"value"})

    df_scores = pd.concat([df_wis,df_mape,df_wis_ratio])
    df_scores['reference_date'] = pd.to_datetime(df_scores['reference_date'])
    df_scores = df_scores[df_scores.Model.isin(models)]
    df_scores = df_scores[df_scores['reference_date']>'2024-09-01']
    return usa_gpd, locations, df_all_forecasts, df_target_data, models,df_scores,scores

# ============================================================================
# NAVIGATION
# ============================================================================

def create_navigation():
    """Create navigation buttons at the top of the page"""
    # Add spacer for Streamlit Cloud deployment bar
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col3:
        # Create button-style navigation
        nav_col1, nav_col2 = st.columns(2)
        
        with nav_col1:
            if st.button("📊 Dashboard", use_container_width=True, 
                        type="primary" if st.session_state.current_page == "Dashboard" else "secondary"):
                st.session_state.current_page = "Dashboard"
                st.rerun()
        
        with nav_col2:
            if st.button("📈 Evaluations", use_container_width=True,
                        type="primary" if st.session_state.current_page == "Evaluations" else "secondary"):
                st.session_state.current_page = "Evaluations"
                st.rerun()

# ============================================================================
# SHARED SIDEBAR
# ============================================================================

def create_sidebar(models, is_evaluations_page=False):
    """Create the shared sidebar for both pages"""
    with st.sidebar:
        st.markdown("### CONFIGURATION")
        
        # Location selection
        st.markdown('<div class="section-header">LOCATION</div>', unsafe_allow_html=True)
        selected_state = st.selectbox(
            "Select State",
            states,
            index=states.index('United States'),
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Model selection
        st.markdown('<div class="section-header">MODEL SELECTION</div>', unsafe_allow_html=True)
        
        selected_models = []
        
        if is_evaluations_page:
            # Single model selection for evaluations page
            selected_model = st.radio(
                "Select Model",
                models,
                format_func=lambda x: x.replace('_', ' '),
                label_visibility="collapsed",
                key="eval_model_selection"
            )
            selected_models = [selected_model]
            
            # Show color indicator for selected model
            color = MODEL_COLORS.get(selected_model, '#808080')
            st.markdown(f'<div style="display: flex; align-items: center;"><div class="model-indicator" style="background-color: {color};"></div><span>{selected_model.replace("_", " ")}</span></div>', 
                       unsafe_allow_html=True)
            
            st.divider()
            
            # Horizon selection for evaluations page
            st.markdown('<div class="section-header">FORECAST HORIZON</div>', unsafe_allow_html=True)
            
            horizons = [0, 1, 2, 3]
            horizon_cols = st.columns(4)
            
            for i, (col, h) in enumerate(zip(horizon_cols, horizons)):
                with col:
                    if st.button(
                        str(h),
                        key=f"horizon_{h}",
                        help=f"{'Nowcast' if h == 0 else f'{h} week{"s" if h != 1 else ""} ahead'}",
                        use_container_width=True,
                        type="primary" if st.session_state.selected_horizon == h else "secondary"
                    ):
                        st.session_state.selected_horizon = h
                        st.rerun()
            
            st.caption(f"Selected: {'Nowcast' if st.session_state.selected_horizon == 0 else f'{st.session_state.selected_horizon} week{"s" if st.session_state.selected_horizon != 1 else ""} ahead'}")
            
        else:
            # Multiple model selection for dashboard
            # Initialize model defaults in session state if not present
            if 'model_defaults' not in st.session_state:
                st.session_state.model_defaults = {model: True for model in models}
            
            # Quick actions for model selection
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All", use_container_width=True):
                    st.session_state.model_defaults = {model: True for model in models}
                    st.rerun()
            with col2:
                if st.button("Clear All", use_container_width=True):
                    st.session_state.model_defaults = {model: False for model in models}
                    st.rerun()
            
            # Create checkboxes for each model
            for model in models:
                model_display = model.replace('_', ' ')
                color = MODEL_COLORS.get(model, '#808080')
                
                # Display model with color indicator
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.markdown(f'<div class="model-indicator" style="background-color: {color};"></div>', 
                               unsafe_allow_html=True)
                with col2:
                    default_value = st.session_state.model_defaults.get(model, True)
                    if st.checkbox(model_display, value=default_value, key=f"model_{model}"):
                        selected_models.append(model)
                        st.session_state.model_defaults[model] = True
                    else:
                        st.session_state.model_defaults[model] = False
        
        st.divider()
        
        # Theme toggle at bottom
        st.markdown('<div class="section-header">APPEARANCE</div>', unsafe_allow_html=True)
        
        theme_toggle = st.toggle(
            "Dark Mode",
            value=(st.session_state.theme == 'dark'),
            help="Switch between light and dark themes"
        )
        
        if theme_toggle and st.session_state.theme == 'light':
            st.session_state.theme = 'dark'
            st.rerun()
        elif not theme_toggle and st.session_state.theme == 'dark':
            st.session_state.theme = 'light'
            st.rerun()
    
    return selected_state, selected_models


def create_sidebar_evals(models, is_evaluations_page=False):
    """Create the shared sidebar for both pages"""
    with st.sidebar:
        st.markdown("### CONFIGURATION")
        
        # Location selection
        st.markdown('<div class="section-header">LOCATION</div>', unsafe_allow_html=True)
        selected_state = st.selectbox(
            "Select State",
            states,
            index=states.index('United States'),
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Model selection
        st.markdown('<div class="section-header">MODEL SELECTION</div>', unsafe_allow_html=True)
        
        selected_models = []
        
        if is_evaluations_page:
            # Single model selection for evaluations page
            selected_model = st.radio(
                "Select Model",
                models,
                format_func=lambda x: x.replace('_', ' '),
                label_visibility="collapsed",
                key="eval_model_selection"
            )
            selected_models = [selected_model]
            
            # Show color indicator for selected model
            color = MODEL_COLORS.get(selected_model, '#808080')
            st.markdown(f'<div style="display: flex; align-items: center;"><div class="model-indicator" style="background-color: {color};"></div><span>{selected_model.replace("_", " ")}</span></div>', 
                       unsafe_allow_html=True)
            
            st.divider()
            
            # Horizon selection for evaluations page
            st.markdown('<div class="section-header">FORECAST HORIZON</div>', unsafe_allow_html=True)
            
            horizons = [0, 1, 2, 3]
            horizon_cols = st.columns(4)
            
            for i, (col, h) in enumerate(zip(horizon_cols, horizons)):
                with col:
                    if st.button(
                        str(h),
                        key=f"horizon_{h}",
                        help=f"{'Nowcast' if h == 0 else f'{h} week{"s" if h != 1 else ""} ahead'}",
                        use_container_width=True,
                        type="primary" if st.session_state.selected_horizon == h else "secondary"
                    ):
                        st.session_state.selected_horizon = h
                        st.rerun()
            
            st.caption(f"Selected: {'Nowcast' if st.session_state.selected_horizon == 0 else f'{st.session_state.selected_horizon} week{"s" if st.session_state.selected_horizon != 1 else ""} ahead'}")
            
        else:
            # Multiple model selection for dashboard
            # Initialize model defaults in session state if not present
            if 'model_defaults' not in st.session_state:
                st.session_state.model_defaults = {model: True for model in models}
            
            # Quick actions for model selection
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All", use_container_width=True):
                    st.session_state.model_defaults = {model: True for model in models}
                    st.rerun()
            with col2:
                if st.button("Clear All", use_container_width=True):
                    st.session_state.model_defaults = {model: False for model in models}
                    st.rerun()
            
            # Create checkboxes for each model
            for model in models:
                model_display = model.replace('_', ' ')
                color = MODEL_COLORS.get(model, '#808080')
                
                # Display model with color indicator
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.markdown(f'<div class="model-indicator" style="background-color: {color};"></div>', 
                               unsafe_allow_html=True)
                with col2:
                    default_value = st.session_state.model_defaults.get(model, True)
                    if st.checkbox(model_display, value=default_value, key=f"model_{model}"):
                        selected_models.append(model)
                        st.session_state.model_defaults[model] = True
                    else:
                        st.session_state.model_defaults[model] = False
        
        st.divider()
        selected_score = st.sidebar.selectbox(
            "Choose a Score:",
            scores
        )

        
        st.divider()

        
        # Theme toggle at bottom
        st.markdown('<div class="section-header">APPEARANCE</div>', unsafe_allow_html=True)
        
        theme_toggle = st.toggle(
            "Dark Mode",
            value=(st.session_state.theme == 'dark'),
            help="Switch between light and dark themes"
        )
        
        if theme_toggle and st.session_state.theme == 'light':
            st.session_state.theme = 'dark'
            st.rerun()
        elif not theme_toggle and st.session_state.theme == 'dark':
            st.session_state.theme = 'light'
            st.rerun()
    
    return selected_state, selected_models,selected_score

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

def dashboard_page(selected_state, selected_models, locations, df_forecasts, df_target_data, models):
    """Main dashboard page"""
    # Main content header
    st.markdown(f"""
    <div style="
        background: {COLORS['background']};
        padding: 1.5rem 2rem;
        margin: 1rem -1rem 2rem -1rem;
        border-bottom: 2px solid {COLORS['border']};
    ">
        <h1 style="margin: 0; font-size: 1.875rem; color: {COLORS['text']};">
            {selected_state} Influenza Dashboard
        </h1>
        <p style="color: {COLORS['text_muted']}; margin-top: 0.25rem; font-size: 0.875rem;">
            Weekly Hospitalizations Forecast · Updated {datetime.now().strftime('%B %d, %Y')} · {len(selected_models)} model(s) selected
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get forecast data for selected state and models
    if selected_state == "United States":
        location_id = "US"
    else:
        location_id = locations[locations.location_name == selected_state].location.unique()[0]
    df_state_forecast = df_forecasts[(df_forecasts.location == location_id) & 
                                     (df_forecasts.model.isin(selected_models))]
    df_state_target = df_target_data[df_target_data.location == location_id]
    
    # Get unique reference dates
    dates = df_state_forecast['reference_date'].sort_values().unique()
    
    # Date selector
    if len(dates) > 1:
        selected_date_idx = st.select_slider(
            "📅 Select Forecast Date",
            options=range(len(dates)),
            value=len(dates) - 1,
            format_func=lambda x: dates[x].strftime('%B %d, %Y'),
            key="main_date_slider"
        )
    else:
        selected_date_idx = 0
    
    # Main layout
    col1, col2 = st.columns([3, 7])
    
    with col1:
        st.markdown('<div class="section-header">FORECAST METRICS</div>', unsafe_allow_html=True)
        
        if len(selected_models) > 0 and len(dates) > 0:
            selected_ref_date = dates[selected_date_idx]
            
            # Calculate ensemble metrics
            ensemble_metrics = []
            
            for model in selected_models:
                df_model = df_state_forecast[(df_state_forecast['reference_date'] == selected_ref_date) & 
                                            (df_state_forecast['model'] == model)]
                df_median = df_model[df_model['output_type_id'] == 0.5].sort_values('target_end_date')
                
                if len(df_median) > 0:
                    next_week_value = df_median['value'].iloc[0]
                    four_week_avg = df_median['value'].mean()
                    peak_value = df_median['value'].max()
                    
                    df_next_week = df_model[df_model['target_end_date'] == df_median['target_end_date'].iloc[0]]
                    ci_95_lower = df_next_week[df_next_week['output_type_id'] == 0.025]['value'].iloc[0] if len(df_next_week[df_next_week['output_type_id'] == 0.025]) > 0 else 0
                    ci_95_upper = df_next_week[df_next_week['output_type_id'] == 0.975]['value'].iloc[0] if len(df_next_week[df_next_week['output_type_id'] == 0.975]) > 0 else 0
                    
                    ensemble_metrics.append({
                        'model': model,
                        'next_week': next_week_value,
                        'four_week_avg': four_week_avg,
                        'peak': peak_value,
                        'ci_lower': ci_95_lower,
                        'ci_upper': ci_95_upper
                    })
            
            if ensemble_metrics:
                # Calculate ensemble averages
                avg_next_week = np.mean([m['next_week'] for m in ensemble_metrics])
                avg_four_week = np.mean([m['four_week_avg'] for m in ensemble_metrics])
                avg_peak = np.mean([m['peak'] for m in ensemble_metrics])
                avg_ci_lower = np.mean([m['ci_lower'] for m in ensemble_metrics])
                avg_ci_upper = np.mean([m['ci_upper'] for m in ensemble_metrics])
                
                # Get latest observed value
                latest_observed = df_state_target['value'].iloc[-1] if len(df_state_target) > 0 else 0
                
                # Status determination
                if avg_next_week > 2000:
                    status = "HIGH"
                    status_color = COLORS['danger']
                elif avg_next_week > 1000:
                    status = "MODERATE"
                    status_color = COLORS['warning']
                else:
                    status = "LOW"
                    status_color = COLORS['success']
                
                # Display ensemble metrics
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 0.75rem; color: {COLORS['text_muted']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                        Ensemble Forecast ({len(selected_models)} models)
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 600; color: {status_color}; margin-bottom: 1rem;">
                        {status} ACTIVITY
                    </div>
                    <div style="border-top: 1px solid {COLORS['border']}; padding-top: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="color: {COLORS['text_muted']}; font-size: 0.75rem;">Next Week</span>
                            <span style="color: {COLORS['text']}; font-weight: 500; font-size: 0.875rem;">{avg_next_week:,.0f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="color: {COLORS['text_muted']}; font-size: 0.75rem;">95% CI</span>
                            <span style="color: {COLORS['text']}; font-weight: 500; font-size: 0.875rem;">{avg_ci_lower:,.0f} - {avg_ci_upper:,.0f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="color: {COLORS['text_muted']}; font-size: 0.75rem;">4-Week Avg</span>
                            <span style="color: {COLORS['text']}; font-weight: 500; font-size: 0.875rem;">{avg_four_week:,.0f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: {COLORS['text_muted']}; font-size: 0.75rem;">Peak Forecast</span>
                            <span style="color: {COLORS['text']}; font-weight: 500; font-size: 0.875rem;">{avg_peak:,.0f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Last Observed", f"{latest_observed:,.0f}")
                with col_m2:
                    st.metric("Models Active", f"{len(selected_models)}/{len(models)}")
        else:
            st.warning("No models selected or no data available")
    
    with col2:
        st.markdown('<div class="section-header">WEEKLY HOSPITALIZATIONS FORECAST</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Plot each selected model
        if len(selected_models) > 0 and len(dates) > 0:
            selected_ref_date = dates[selected_date_idx]
            
            for i, model in enumerate(selected_models):
                df_model = df_state_forecast[(df_state_forecast['reference_date'] == selected_ref_date) & 
                                            (df_state_forecast['model'] == model)]
                
                if len(df_model) == 0:
                    continue
                    
                pivot_df = df_model.pivot_table(
                    index='target_end_date',
                    columns='output_type_id',
                    values='value',
                    aggfunc='first'
                ).reset_index()
                
                pivot_df = pivot_df.sort_values('target_end_date')
                x_dates = pivot_df['target_end_date'].tolist()
                
                model_color = MODEL_COLORS.get(model, '#808080')
                model_name = model.replace('_', ' ')
                
                # Add confidence intervals
                if 0.025 in pivot_df.columns and 0.975 in pivot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=x_dates + x_dates[::-1],
                        y=pivot_df[0.975].tolist() + pivot_df[0.025].tolist()[::-1],
                        fill='toself',
                        fillcolor=f'rgba({int(model_color[1:3], 16)}, {int(model_color[3:5], 16)}, {int(model_color[5:7], 16)}, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{model_name} 95% CI',
                        legendgroup=model,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Median line
                if 0.5 in pivot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=x_dates,
                        y=pivot_df[0.5].tolist(),
                        mode='lines+markers',
                        name=model_name,
                        legendgroup=model,
                        line=dict(color=model_color, width=2),
                        marker=dict(size=5, color=model_color),
                        hovertemplate=f'<b>{model_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Median: %{{y:,.0f}}<extra></extra>'
                    ))
        
        # Add observed data
        fig.add_trace(go.Scatter(
            x=df_state_target['date'],
            y=df_state_target['value'],
            mode='markers',
            name='Observed',
            marker=dict(
                color=COLORS['danger'],
                size=8,
                symbol='circle',
                line=dict(color=COLORS['danger'], width=1)
            ),
            hovertemplate='<b>Observed</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:,.0f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=None,
            xaxis_title="Date",
            yaxis_title="Weekly Hospitalizations",
            hovermode='x unified',
            template=COLORS['plotly_template'],
            height=500,
            margin=dict(t=40, b=60, l=60, r=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor=COLORS['background'],
                bordercolor=COLORS['border'],
                borderwidth=1,
                font=dict(size=11)
            ),
            paper_bgcolor=COLORS['plotly_bg'],
            plot_bgcolor=COLORS['plotly_bg']
        )
        
        fig.update_xaxes(
            tickformat="%b %d",
            tickangle=-45,
            showgrid=True,
            gridwidth=1,
            gridcolor=COLORS['plotly_grid']
        )
        
        fig.update_yaxes(
            rangemode='tozero',
            tickformat=',',
            showgrid=True,
            gridwidth=1,
            gridcolor=COLORS['plotly_grid']
        )
        
        config = {'displayModeBar': True, 'displaylogo': False}
        st.plotly_chart(fig, use_container_width=True, config=config)

# ============================================================================
# EVALUATIONS PAGE
# ============================================================================

def evaluations_page(selected_state, selected_models,selected_score, locations, df_forecasts, df_target_data,df_scores):
    """Evaluations page showing forecast performance across horizons"""
    
    selected_horizon = st.session_state.selected_horizon
    
    # Main content header
    st.markdown(f"""
    <div style="
        background: {COLORS['background']};
        padding: 1.5rem 2rem;
        margin: 1rem -1rem 2rem -1rem;
        border-bottom: 2px solid {COLORS['border']};
    ">
        <h1 style="margin: 0; font-size: 1.875rem; color: {COLORS['text']};">
            Forecast Evaluations - {selected_state}
        </h1>
        <p style="color: {COLORS['text_muted']}; margin-top: 0.25rem; font-size: 0.875rem;">
            Model: {selected_models[0].replace('_', ' ')} · Horizon {selected_horizon} · Updated {datetime.now().strftime('%B %d, %Y')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data for selected state and model
    if selected_state=="United States":
        location_id = "US"
    else:
        location_id = locations[locations.location_name == selected_state].location.unique()[0]

    selected_model = selected_models[0]
    
    # Get all forecasts for the model and location
    df_model_all = df_forecasts[(df_forecasts.location == location_id) & 
                                (df_forecasts.model == selected_model)]
    df_scores_all = df_scores[(df_scores.location == location_id) & 
                                (df_scores.Model == selected_model)]
    
    # Filter for selected horizon
    df_horizon = df_model_all[df_model_all['horizon'] == selected_horizon].copy()
    df_horizon_score = df_scores_all[df_scores_all['horizon'] == selected_horizon].copy()
    df_horizon_score['target_end_date'] = pd.to_datetime(df_horizon_score['target_end_date'])
    df_horizon_score = df_horizon_score[df_horizon_score.score==selected_score]
    # Get observed data
    df_state_target = df_target_data[df_target_data.location == location_id]
    df_horizon_score.sort_values(by = "target_end_date", inplace = True)
    # Main visualization - Simple 95% CI Boxes (COLORED BY MODEL)
    st.markdown('<div class="section-header">HOSPITALIZATION FORECASTS BY HORIZON</div>', unsafe_allow_html=True)
    
    # Create the visualization
    fig = go.Figure()
    
    if len(df_horizon) > 0:
        # Get model color
        model_color = MODEL_COLORS.get(selected_model, '#808080')
        # Convert hex to RGB for transparency
        r = int(model_color[1:3], 16)
        g = int(model_color[3:5], 16)
        b = int(model_color[5:7], 16)
        
        # Get all unique reference dates
        reference_dates = sorted(df_horizon['reference_date'].unique())
        
        # For each reference date, create boxes for the 95% CI
        for ref_date in reference_dates:  # Show last 20 forecast dates
            df_ref = df_horizon[df_horizon['reference_date'] == ref_date]
            
            # Pivot to get quantiles by target date
            pivot_df = df_ref.pivot_table(
                index='target_end_date',
                columns='output_type_id',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            if len(pivot_df) == 0:
                continue
            
            # Check if we have the needed quantiles for 95% CI
            if 0.025 in pivot_df.columns and 0.975 in pivot_df.columns and 0.5 in pivot_df.columns:
                # Calculate days since reference for opacity
                days_old = (reference_dates[-1] - ref_date).days
                opacity = 0.4#max(0.2, min(0.6, 1 - (days_old / 60)))
                
                # Color based on model with recency-based opacity
                box_color = f'rgba({r}, {g}, {b}, {opacity * 0.4})'
                line_color = f'rgba({r}, {g}, {b}, {opacity + 0.2})'
                
                for idx, row in pivot_df.iterrows():
                    target_date = row['target_end_date']
                    lower_95 = row[0.025]
                    upper_95 = row[0.975]
                    median = row[0.5]
                    
                    # Add rectangle for 95% CI
                    fig.add_shape(
                        type="rect",
                        x0=target_date - timedelta(days=2),
                        x1=target_date + timedelta(days=2),
                        y0=lower_95,
                        y1=upper_95,
                        fillcolor=box_color,
                        line=dict(color=line_color, width=1),
                        layer="below"
                    )
                    
                    # Add median line
                    fig.add_trace(go.Scatter(
                        x=[target_date - timedelta(days=2), target_date + timedelta(days=2)],
                        y=[median, median],
                        mode='lines',
                        line=dict(color='black', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # Add observed data as dots
    fig.add_trace(go.Scatter(
        x=df_state_target['date'],
        y=df_state_target['value'],
        mode='markers',
        name='Observed',
        marker=dict(
            color='white',
            size=8,
            symbol='circle',
            line=dict(color='black', width=2)
        ),
        hovertemplate='<b>Observed</b><br>Date: %{x|%b %d}<br>Value: %{y:,.0f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"95% Confidence Intervals - Horizon {selected_horizon}",
        xaxis_title="Target Date",
        yaxis_title="Weekly Hospitalizations",
        template=COLORS['plotly_template'],
        height=500,
        showlegend=True,
        paper_bgcolor=COLORS['plotly_bg'],
        plot_bgcolor=COLORS['plotly_bg'],
        hovermode='x unified'
    )
    
    # Format axes
    fig.update_xaxes(
        tickformat="%b %d",
        showgrid=True,
        gridwidth=0.5,
        gridcolor=COLORS['plotly_grid']
    )
    
    fig.update_yaxes(
        tickformat=',',
        showgrid=True,
        gridwidth=0.5,
        gridcolor=COLORS['plotly_grid'],
        rangemode='tozero'
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
    
    # 2 Main visualization - Simple 95% CI Boxes (COLORED BY MODEL)
    st.markdown('<div class="section-header">Score FORECASTS BY HORIZON</div>', unsafe_allow_html=True)
    
    # Create the visualization
    fig_score = go.Figure()

    # Add observed data as dots
    fig_score.add_trace(go.Scatter(
        x=df_horizon_score['target_end_date'],
        y=df_horizon_score['value'],
        name='Score',
        line=dict(color=model_color, width=3),  # controls the line
        marker=dict(
            color=model_color,
            size=8,
            symbol='circle',
            line=dict(color=model_color, width=7)
        ),
        hovertemplate='skip'
    ))


        # Update layout
    fig_score.update_layout(
        title=f"{selected_score} by horizon",
        xaxis_title="Target Date",
        yaxis_title=f"{selected_score}",
        template=COLORS['plotly_template'],
        height=500,
        showlegend=True,
        paper_bgcolor=COLORS['plotly_bg'],
        plot_bgcolor=COLORS['plotly_bg'],
        hovermode='x unified'
    )
    
    # Format axes
    fig_score.update_xaxes(
        tickformat="%b %d",
        showgrid=True,
        gridwidth=0.5,
        gridcolor=COLORS['plotly_grid'],
        range = [df_target_data.date.min(),df_horizon.target_end_date.max()]
    )
    
    fig_score.update_yaxes(
        tickformat=',',
        showgrid=True,
        gridwidth=0.5,
        gridcolor=COLORS['plotly_grid'],
        rangemode='tozero'
    )
    
    st.plotly_chart(fig_score, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Apply theme
apply_theme_styles()

# Load data
usa_gpd, locations, df_forecasts, df_target_data, models,df_scores,scores = load_data()

# Create navigation at the top
create_navigation()

# Display the appropriate page
if st.session_state.current_page == "Dashboard":
    selected_state, selected_models = create_sidebar(models, is_evaluations_page=False)
    dashboard_page(selected_state, selected_models, locations, df_forecasts, df_target_data, models)
else:
    selected_state, selected_models,selected_score = create_sidebar_evals(models, is_evaluations_page=True)
    evaluations_page(selected_state, selected_models,selected_score, locations, df_forecasts, df_target_data,df_scores)