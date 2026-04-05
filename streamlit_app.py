import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Maritime Vessel Anomaly Detector",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Maritime Vessel Anomaly Detector")
st.markdown("""
*Built by a Maritime Professional transitioning to AI Engineering.*
This dashboard detects suspicious vessel behavior using an **Isolation Forest** model trained on real US Coast Guard AIS data.
""")

# 2. Data Loading
@st.cache_data
@st.cache_data
def load_data():
    # Χρησιμοποιούμε την απόλυτη διαδρομή που μου έδωσες
    file_path = r"C:\Users\jimre\data\processed\vessel_scores_01_01.parquet"
    
    if not os.path.exists(file_path):
        st.error(f"Το αρχείο δεν βρέθηκε στο: {file_path}")
        return pd.DataFrame()
    return pd.read_parquet(file_path)

df = load_data()

if not df.empty:
    # 3. Sidebar Filters
    st.sidebar.header("Control Panel")
    
    # Filter by Anomaly
    show_anomalies_only = st.sidebar.toggle("Show Anomalies Only", value=False)
    
    # Filter by Vessel Type
    v_types = ["All"] + sorted(df['vessel_type'].unique().tolist())
    selected_type = st.sidebar.selectbox("Filter by Vessel Type", v_types)

    # Apply Logic
    display_df = df.copy()
    if show_anomalies_only:
        display_df = display_df[display_df['anomaly_label'] == -1]
    if selected_type != "All":
        display_df = display_df[display_df['vessel_type'] == selected_type]

    # 4. KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_vessels = len(df)
    anomalies_count = (df['anomaly_label'] == -1).sum()
    
    col1.metric("Total Vessels", f"{total_vessels:,}")
    col2.metric("Flagged Anomalies", f"{anomalies_count:,}")
    col3.metric("Anomaly Rate", f"{(anomalies_count/total_vessels)*100:.1f}%")
    col4.metric("Vessels in View", f"{len(display_df):,}")

    st.divider()

    # 5. Main Layout: Map and Analysis
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Interactive Vessel Map")
        st.caption("Red = Anomalous | Blue = Normal behavior")
        
        # Center map on the average coordinates
        m = folium.Map(
            location=[display_df['latitude'].mean(), display_df['longitude'].mean()],
            zoom_start=6,
            tiles='CartoDB positron'
        )

        # Plot vessels
        for _, row in display_df.iterrows():
            is_anomaly = row['anomaly_label'] == -1
            color = 'red' if is_anomaly else '#1f77b4'
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=7 if is_anomaly else 3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(f"""
                    <b>Vessel:</b> {row['vessel_name']}<br>
                    <b>MMSI:</b> {row['mmsi']}<br>
                    <b>Status:</b> {'ANOMALY' if is_anomaly else 'Normal'}<br>
                    <b>Anomaly Score:</b> {row['anomaly_score']:.3f}<br>
                    <b>Dark Periods:</b> {int(row['ais_dark_count'])}
                """, max_width=250)
            ).add_to(m)

        st_folium(m, width="100%", height=600)

    with right_col:
        st.subheader("Top Anomalous Vessels")
        top_anoms = (
            df[df['anomaly_label'] == -1]
            .sort_values('anomaly_score', ascending=True)
            .head(10)
            [['vessel_name', 'anomaly_score', 'ais_dark_count', 'sog_max']]
        )
        st.dataframe(top_anoms, use_container_width=True, hide_index=True)

        st.subheader("Score Distribution")
        fig = px.histogram(
            df, 
            x="anomaly_score", 
            color="anomaly_label",
            color_discrete_map={-1: "red", 1: "#1f77b4"},
            labels={"anomaly_label": "Type (-1=Anomaly)"},
            nbins=50
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Awaiting data processing...")