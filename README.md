# 🚢 Maritime Vessel Anomaly Detector 

## From Bridge to Bytes: A Domain-Driven AI Project
Leveraging my professional background in the maritime industry, I developed this tool to enhance situational awareness at sea. This project isn't just about code; it's about solving real-world safety and security challenges using Machine Learning.

## The Mission
Modern maritime safety relies on AIS data, but manually spotting suspicious behavior is nearly impossible. This application uses **Unsupervised Learning (Isolation Forest)** to automatically flag:
- **Dark Periods:** Potential transponder tampering.
- **Navigational Mismatches:** Discrepancies between reported status and actual movement.
- **Abnormal Maneuvers:** Sudden course/speed changes indicative of risk.

## Technical Implementation
- **Machine Learning:** Scikit-Learn (Isolation Forest) for anomaly detection.
- **Data Engineering:** Python & Pandas for processing 1M+ rows of AIS data.
- **Interactive Dashboard:** Streamlit with Folium maps for real-time visualization.
- **Efficiency:** Parquet data format for high-speed data loading.

## How to Run
1. Clone this repository.
2. Install requirements: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run streamlit_app.py`
