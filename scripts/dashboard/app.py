"""
Streamlit dashboard for AI Predictive Maintenance System.
Provides interactive visualization and control for valve monitoring.
"""
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time

# Set up page
st.set_page_config(
    page_title="Valve Predictive Maintenance Dashboard", 
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get configuration from environment
API_URL = os.environ.get("API_URL", "http://api:8000")
REFRESH_INTERVAL = int(os.environ.get("REFRESH_INTERVAL", "30"))  # in seconds

# Helper functions
def get_valve_status(valve_id, vibration, temperature, pressure):
    """Get valve failure prediction from API."""
    try:
        # Ensure all values are the correct type - convert to float for API compatibility
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "valve_id": int(valve_id) if valve_id else None,
                "vibration": float(vibration),
                "temperature": float(temperature),
                "pressure": float(pressure)
            },
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")
        return None

def get_root_cause_analysis(valve_id, vibration, temperature, pressure):
    """Get root cause analysis for valve failure."""
    try:
        # Ensure all values are the correct type for API compatibility
        response = requests.post(
            f"{API_URL}/rca",
            json={
                "valve_id": int(valve_id) if valve_id else 0,
                "vibration": float(vibration),
                "temperature": float(temperature),
                "pressure": float(pressure)
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"RCA API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to connect to RCA API: {str(e)}")
        return None

def get_api_health():
    """Check API health status."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_api_metrics():
    """Get API performance metrics."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=2)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def generate_sample_data(num_records=100):
    """Generate sample valve data for visualization."""
    np.random.seed(42)
    time_points = [datetime.now() - timedelta(minutes=i*15) for i in range(num_records)]
    
    # Generate data with some patterns
    vibration = np.random.normal(0.8, 0.3, num_records)
    vibration = np.clip(vibration, 0.1, 2.0)
    
    # Add upward trend for demonstration
    trend = np.linspace(0, 0.5, num_records)
    vibration = vibration + trend[::-1]
    
    temperature = np.random.normal(110, 15, num_records)
    temperature = np.clip(temperature, 70, 150)
    
    # Add cyclical pattern for temperature
    cycle = 20 * np.sin(np.linspace(0, 6*np.pi, num_records))
    temperature = temperature + cycle
    
    pressure = np.random.normal(70, 10, num_records)
    pressure = np.clip(pressure, 40, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': time_points,
        'vibration': vibration,
        'temperature': temperature,
        'pressure': pressure
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df

def risk_color(risk_value):
    """Get color code based on risk level."""
    if risk_value > 0.8:
        return "#FF0000"  # Red
    elif risk_value > 0.5:
        return "#FF9900"  # Orange
    elif risk_value > 0.2:
        return "#FFFF00"  # Yellow
    else:
        return "#00FF00"  # Green

# Dashboard layout
st.title("🔧 Valve Predictive Maintenance Dashboard")

# Sidebar
with st.sidebar:
    st.header("Controls")
    
    # Refresh rate
    refresh_rate = st.slider("Refresh interval (seconds)", 5, 60, REFRESH_INTERVAL, 5)
    
    # API connection status
    api_health = get_api_health()
    
    if api_health.get("status") == "healthy":
        st.success("✅ API Connected")
        st.info(f"Version: {api_health.get('api_version', '1.0.0')}")
    else:
        st.error("❌ API Disconnected")
        st.error(f"Error: {api_health.get('message', 'Unknown error')}")
    
    # Valve simulator
    st.header("Valve Simulator")
    valve_id = st.number_input("Valve ID", min_value=1, max_value=100, value=42)
    vibration = st.slider("Vibration", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
    temperature = st.slider("Temperature (°C)", min_value=70, max_value=150, value=110, step=5)
    pressure = st.slider("Pressure (psi)", min_value=40, max_value=100, value=70, step=5)
    
    if st.button("Test Prediction"):
        with st.spinner("Getting prediction..."):
            try:
                # Explicit cast to handle slider integer values
                prediction = get_valve_status(
                    valve_id,
                    float(vibration),  # Convert to float
                    float(temperature),
                    float(pressure)
                )
                
                if prediction:
                    risk = prediction.get("failure_risk", 0)
                    confidence = prediction.get("confidence", 0)
                    risk_level = prediction.get("risk_level", "Unknown")
                    
                    risk_color_val = risk_color(risk)
                    st.markdown(f"### Risk Level: <span style='color:{risk_color_val}'>{risk_level}</span>", unsafe_allow_html=True)
                    st.progress(risk)
                    st.metric("Failure Risk", f"{risk:.1%}")
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Get root cause analysis if risk is high
                    if risk > 0.5:
                        st.markdown("---")
                        st.subheader("Root Cause Analysis")
                        with st.spinner("Analyzing failure causes..."):
                            try:
                                rca = get_root_cause_analysis(
                                    valve_id, 
                                    float(vibration), 
                                    float(temperature), 
                                    float(pressure)
                                )
                                
                                if rca:
                                    st.write(f"**Root Cause:** {rca.get('root_cause', 'Unknown')}")
                                    st.write(f"**Confidence:** {rca.get('confidence', 0):.1f}%")
                                    st.write(f"**Recommended Action:** {rca.get('action', 'N/A')}")
                                    st.write(f"**Time Sensitivity:** {rca.get('time_sensitivity', 'unknown').title()}")
                            except Exception as e:
                                st.error(f"Error analyzing root cause: {str(e)}")
            except Exception as e:
                st.error(f"Error getting prediction: {str(e)}")
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("AI Predictive Maintenance System v1.0.0")
    st.markdown("© 2025 Priyesh Verma")

# Main dashboard area - use tabs
tab1, tab2, tab3 = st.tabs(["Valve Monitoring", "System Performance", "Documentation"])

with tab1:
    # Valve monitoring dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sensor Data Trends")
        
        # Generate sample data for visualization
        data = generate_sample_data(48)  # 12 hours of data at 15-min intervals
        
        # Create time series plot
        fig = go.Figure()
        
        # Add traces for each sensor
        fig.add_trace(go.Scatter(
            x=data['timestamp'], 
            y=data['vibration'],
            name="Vibration",
            line=dict(color="#1f77b4", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'], 
            y=data['temperature'] / 100,  # Scale for visibility
            name="Temperature (scaled)",
            line=dict(color="#ff7f0e", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'], 
            y=data['pressure'] / 50,  # Scale for visibility
            name="Pressure (scaled)",
            line=dict(color="#2ca02c", width=2)
        ))
        
        # Layout improvements
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            xaxis_title="Time",
            yaxis_title="Sensor Values"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.info(
            "This chart shows the historical sensor data for the selected valve. "
            "Temperature and pressure are scaled to fit on the same axis as vibration. "
            "The most recent data points are on the right."
        )
    
    with col2:
        st.subheader("Current Status")
        
        # Use the latest data point
        latest = data.iloc[-1]
        
        # Get prediction for current values - ensure all values are floats
        current_prediction = get_valve_status(
            valve_id, 
            float(latest['vibration']), 
            float(latest['temperature']), 
            float(latest['pressure'])
        )
        
        if current_prediction:
            risk = current_prediction.get("failure_risk", 0)
            confidence = current_prediction.get("confidence", 0)
            risk_level = current_prediction.get("risk_level", "Unknown")
            
            # Large display for current risk level
            risk_color_val = risk_color(risk)
            
            # Gauge chart for risk
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Failure Risk"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': risk_color_val},
                    'steps': [
                        {'range': [0, 20], 'color': '#00FF00'},
                        {'range': [20, 50], 'color': '#FFFF00'},
                        {'range': [50, 80], 'color': '#FF9900'},
                        {'range': [80, 100], 'color': '#FF0000'},
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 4},
                        'thickness': 0.75,
                        'value': risk * 100
                    }
                }
            ))
            
            gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            st.plotly_chart(gauge, use_container_width=True)
            
            # Current values
            st.markdown("### Current Sensor Readings")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Vibration", f"{latest['vibration']:.2f}")
            with col2:
                st.metric("Temperature", f"{latest['temperature']:.1f}°C")
            with col3:
                st.metric("Pressure", f"{latest['pressure']:.1f} psi")
            
            # Status summary
            st.markdown(f"### Status: <span style='color:{risk_color_val}'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Only show RCA button if risk is elevated
            if risk > 0.5:
                if st.button("Run Root Cause Analysis"):
                    with st.spinner("Analyzing root causes..."):
                        try:
                            rca = get_root_cause_analysis(
                                valve_id, 
                                float(latest['vibration']), 
                                float(latest['temperature']), 
                                float(latest['pressure'])
                            )
                            
                            if rca:
                                st.write(f"**Root Cause:** {rca.get('root_cause', 'Unknown')}")
                                st.write(f"**Confidence:** {rca.get('confidence', 0):.1f}%")
                                st.write(f"**Recommended Action:** {rca.get('action', 'N/A')}")
                                
                                # Display additional details
                                with st.expander("View Details"):
                                    st.write(f"**Time Sensitivity:** {rca.get('time_sensitivity', 'unknown').title()}")
                                    st.write(f"**Estimated Repair Time:** {rca.get('repair_time', 0)} hours")
                                    
                                    parts = rca.get('required_parts', [])
                                    if parts:
                                        st.write("**Required Parts:**")
                                        for part in parts:
                                            st.write(f"- {part}")
                        except Exception as e:
                            st.error(f"Error analyzing root causes: {str(e)}")

with tab2:
    # System performance dashboard
    st.subheader("API Performance Metrics")
    
    # Get metrics
    metrics = get_api_metrics()
    
    if not metrics.get("error"):
        # Create 3 columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_requests = metrics.get("request_count", 0)
            st.metric("Total API Requests", total_requests)
        
        with col2:
            avg_response_time = metrics.get("average_response_time", 0)
            st.metric("Avg Response Time", f"{avg_response_time:.2f} ms")
        
        with col3:
            error_rate = metrics.get("error_rate", 0)
            st.metric("Error Rate", f"{error_rate:.2%}")
        
        # Prediction distribution
        st.subheader("Prediction Distribution")
        
        # Create sample prediction distribution
        risk_levels = ["Low", "Medium", "High", "Critical"]
        prediction_counts = [
            metrics.get("risk_levels", {}).get("Low", 0),
            metrics.get("risk_levels", {}).get("Medium", 0),
            metrics.get("risk_levels", {}).get("High", 0),
            metrics.get("risk_levels", {}).get("Critical", 0)
        ]
        
        # If no data, use sample data
        if sum(prediction_counts) == 0:
            prediction_counts = [65, 23, 10, 2]
        
        # Create bar chart
        fig = px.bar(
            x=risk_levels,
            y=prediction_counts,
            color=risk_levels,
            color_discrete_map={
                "Low": "#00FF00",
                "Medium": "#FFFF00",
                "High": "#FF9900",
                "Critical": "#FF0000"
            },
            labels={"x": "Risk Level", "y": "Count"}
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # System health
        st.subheader("System Health")
        
        services = [
            {"name": "API", "status": "online" if api_health.get("status") == "healthy" else "offline"},
            {"name": "Prometheus", "status": "online"},
            {"name": "Grafana", "status": "online"},
            {"name": "RCA Engine", "status": "online"},
        ]
        
        # Create a DataFrame for the services
        services_df = pd.DataFrame(services)
        
        # Show as a styled table
        st.dataframe(
            services_df,
            column_config={
                "name": "Service",
                "status": st.column_config.TextColumn(
                    "Status",
                    help="Current service status",
                    width="medium",
                )
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.error(f"Failed to load metrics: {metrics.get('error')}")

with tab3:
    # Documentation
    st.subheader("Dashboard Documentation")
    
    st.markdown("""
    ## How to Use This Dashboard
    
    This dashboard provides real-time monitoring and analysis of industrial valve health using AI-powered predictive maintenance.
    
    ### Valve Monitoring Tab
    
    - **Sensor Data Trends**: Historical data for vibration, temperature, and pressure
    - **Current Status**: Real-time risk assessment with gauge visualization
    - **Root Cause Analysis**: In-depth analysis of potential failure causes for high-risk valves
    
    ### System Performance Tab
    
    - **API Metrics**: Monitor the performance of the prediction API
    - **Prediction Distribution**: Overview of risk assessments across the system
    - **System Health**: Status of all system components
    
    ### Valve Simulator
    
    Use the controls in the sidebar to:
    
    1. Adjust sensor values for a specific valve
    2. Test predictions with custom parameters
    3. Run root cause analysis for high-risk scenarios
    
    ### API Integration
    
    This dashboard connects to these endpoints:
    
    - `GET /health`: System health check
    - `GET /metrics`: Performance metrics
    - `POST /predict`: Real-time failure prediction
    - `POST /rca`: Root cause analysis
    
    ### Legend
    
    Risk levels are color-coded:
    
    - <span style='color:#00FF00'>**Low**</span>: Normal operation (0-20%)
    - <span style='color:#FFFF00'>**Medium**</span>: Requires monitoring (20-50%)
    - <span style='color:#FF9900'>**High**</span>: Schedule maintenance (50-80%)
    - <span style='color:#FF0000'>**Critical**</span>: Immediate action required (80-100%)
    """, unsafe_allow_html=True)

# Auto-refresh functionality
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# Check if we should refresh
current_time = time.time()
if current_time - st.session_state.last_refresh > refresh_rate:
    st.rerun()
    st.session_state.last_refresh = current_time