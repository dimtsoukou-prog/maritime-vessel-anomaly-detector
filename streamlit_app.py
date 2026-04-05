import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os

st.set_page_config(
    page_title="Maritime Vessel Anomaly Detector",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Maritime Vessel Anomaly Detector")
st.markdown("""
*Built by a Maritime Professional transitioning to AI Engineering.*  
This dashboard detects suspicious vessel behaviour using an **Isolation Forest** model trained on real US Coast Guard AIS data.
""")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, 'data', 'processed', 'vessel_scores_01_01.parquet')
    return pd.read_parquet(path)

df = load_data()

if not df.empty:
    st.sidebar.header("Control Panel")
    show_anomalies_only = st.sidebar.toggle("Show Anomalies Only", value=False)
    v_types = ["All"] + sorted(df['vessel_type'].dropna().unique().tolist())
    selected_type = st.sidebar.selectbox("Filter by Vessel Type", v_types)

    display_df = df.copy()
    if show_anomalies_only:
        display_df = display_df[display_df['anomaly_label'] == -1]
    if selected_type != "All":
        display_df = display_df[display_df['vessel_type'] == selected_type]

    col1, col2, col3, col4 = st.columns(4)
    total_vessels = len(df)
    anomalies_count = (df['anomaly_label'] == -1).sum()
    col1.metric("Total Vessels",     f"{total_vessels:,}")
    col2.metric("Flagged Anomalies", f"{anomalies_count:,}")
    col3.metric("Anomaly Rate",      f"{anomalies_count / total_vessels * 100:.1f}%")
    col4.metric("Vessels in View",   f"{len(display_df):,}")

    st.divider()

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Interactive Vessel Map")
        st.caption("Red = Anomalous | Blue = Normal behaviour")

        map_center_lat = display_df['latitude'].mean()
        map_center_lon = display_df['longitude'].mean()

        m = folium.Map(
            location=[map_center_lat, map_center_lon],
            zoom_start=6,
            tiles='CartoDB positron'
        )

        for _, row in display_df.iterrows():
            is_anomaly = row['anomaly_label'] == -1
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=7 if is_anomaly else 3,
                color='red' if is_anomaly else '#1f77b4',
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"""
                    <b>Vessel:</b> {row['vessel_name']}<br>
                    <b>MMSI:</b> {row['mmsi']}<br>
                    <b>Status:</b> {'ANOMALY' if is_anomaly else 'Normal'}<br>
                    <b>Anomaly Score:</b> {row['anomaly_score']:.3f}<br>
                    <b>Dark Periods:</b> {int(row['ais_dark_count'])}<br>
                    <b>Max Speed:</b> {row['sog_max']:.1f} kts<br>
                    <b>Identity Score:</b> {row['identity_completeness']:.2f}
                    """,
                    max_width=250
                )
            ).add_to(m)

        st_folium(m, width="100%", height=600)

    with right_col:
        st.subheader("Top Anomalous Vessels")
        top_anoms = (
            df[df['anomaly_label'] == -1]
            .sort_values('anomaly_score', ascending=True)
            .head(10)
            [['vessel_name', 'anomaly_score', 'ais_dark_count',
              'sog_max', 'identity_completeness']]
            .rename(columns={
                'vessel_name':           'Vessel',
                'anomaly_score':         'Score',
                'ais_dark_count':        'Dark periods',
                'sog_max':               'Max speed (kts)',
                'identity_completeness': 'Identity'
            })
        )
        top_anoms['Score']    = top_anoms['Score'].round(3)
        top_anoms['Identity'] = top_anoms['Identity'].round(2)
        st.dataframe(top_anoms, use_container_width=True, hide_index=True)

        st.subheader("Score Distribution")
        fig = px.histogram(
            df,
            x='anomaly_score',
            color=df['anomaly_label'].map({-1: 'Anomaly', 1: 'Normal'}),
            color_discrete_map={'Anomaly': 'red', 'Normal': '#1f77b4'},
            nbins=50,
            labels={'anomaly_score': 'Anomaly score', 'color': ''}
        )
        fig.update_layout(showlegend=True, height=300,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Awaiting data...")
