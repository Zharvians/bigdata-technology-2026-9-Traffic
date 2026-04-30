# ================================================
# DASHBOARD STREAMLIT - (ENHANCED UAS VERSION)
# ================================================

import streamlit as st
from pyspark.sql import SparkSession
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

# ================================================
# CONFIG
# ================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

st.set_page_config(
    page_title="Smart Traffic Dashboard",
    layout="wide"
)

st.title("🚦 Smart City Traffic Dashboard")
st.markdown(" Simulasi Analisis kemacetan di 10 kota Indonesia secara real-time by Zharvian")

# ================================================
# STYLE (BIAR KEREN)
# ================================================
st.markdown("""
<style>
.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================================================
# INIT SPARK
# ================================================
@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("Dashboard_App").getOrCreate()

spark = get_spark()

# ================================================
# LOAD DATA
# ================================================
def load_parquet(folder_name):
    path = os.path.join(OUTPUT_DIR, folder_name)
    if not os.path.exists(path):
        st.error(f"⚠️ Data '{folder_name}' tidak ditemukan. Jalankan script utama dulu.")
        st.stop()
    return spark.read.parquet(path).toPandas()

pdf = load_parquet("traffic_summary")
pdf_time = load_parquet("traffic_trend")
pdf_ml = load_parquet("ml_ready")

# ================================================
# SIDEBAR FILTER
# ================================================
st.sidebar.header("🔍 Filter")

locations = pdf["location"].unique()
selected_locs = st.sidebar.multiselect(
    "Pilih Kota",
    locations,
    default=locations[:3]
)

filtered_pdf = pdf[pdf["location"].isin(selected_locs)]

# ================================================
# KPI SECTION
# ================================================
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Kendaraan Nasional", int(pdf["total_vehicle"].sum()))

with col2:
    top_city = pdf.sort_values("total_vehicle", ascending=False).iloc[0]
    st.metric("Kota Terpadat", top_city["location"])

with col3:
    st.metric("Status Kota Terpadat", top_city["traffic_status"])

# ================================================
# BAR CHART (RANKING KOTA)
# ================================================
st.markdown("---")
st.subheader("🏙️ Ranking Kemacetan Kota")

fig_bar = px.bar(
    filtered_pdf.sort_values("total_vehicle", ascending=False),
    x="location",
    y="total_vehicle",
    color="traffic_status",
    text="total_vehicle"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ================================================
# TIME SERIES
# ================================================
st.subheader("📈 Traffic Trend (Per 10 Menit)")

pdf_time["start_time"] = pdf_time["window"].apply(
    lambda x: x[0] if isinstance(x, tuple) else x.start
)

pdf_time_filtered = pdf_time[pdf_time["location"].isin(selected_locs)]

fig_line = px.line(
    pdf_time_filtered,
    x="start_time",
    y="total_vehicle",
    color="location"
)

st.plotly_chart(fig_line, use_container_width=True)

# ================================================
# AI PREDICTION
# ================================================
st.markdown("---")
st.subheader("🤖 AI Prediction (Per Kota)")

selected_city_ai = st.selectbox("Pilih Kota untuk Prediksi", locations)

city_data = pdf_ml[pdf_ml["location"] == selected_city_ai]

X = city_data[["hour"]]
y = city_data["vehicle_count"]

model = LinearRegression()
model.fit(X, y)

hour_input = st.slider("Prediksi Jam", 0, 23, 12)

pred = model.predict([[hour_input]])

st.success(
    f"🚗 Prediksi kendaraan di {selected_city_ai} jam {hour_input}:00 ≈ {int(pred[0])}"
)

# ================================================
# FOOTER
# ================================================
st.markdown("---")
st.caption("Made with PySpark + Streamlit + Muhammad Ade Ramadhani 🚀")