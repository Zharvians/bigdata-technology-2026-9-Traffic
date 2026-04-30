# =====================================================
# MAIN UTS BIG DATA - (ENHANCED VERSION)
# =====================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, window, sum as _sum, hour, desc
import random
from datetime import datetime, timedelta
import os
import shutil

# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# =====================================================
# INIT SPARK
# =====================================================

spark = SparkSession.builder \
    .appName("UTS BigData Smart Traffic") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("🚀 Spark Ready - Smart Traffic Analysis Dimulai...")

# =====================================================
# PREPARE OUTPUT FOLDER
# =====================================================

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# GENERATE SMART DUMMY DATA (10 KOTA)
# =====================================================

locations = [
    "Depok", "Jakarta", "Balikpapan", "Banjarmasin", 
    "Banjarbaru", "Medan", "Aceh", "Jayapura", 
    "Makassar", "Bekasi"
]

start_time = datetime(2026, 1, 1, 6, 0)

sensor_data = []

for i in range(300):  # lebih banyak data (biar keliatan big data vibes)
    for loc in locations:
        current_time = start_time + timedelta(minutes=i)
        hour_now = current_time.hour

        # Simulasi jam sibuk (rush hour)
        if 7 <= hour_now <= 9 or 16 <= hour_now <= 19:
            vehicle = random.randint(80, 150)
        else:
            vehicle = random.randint(10, 70)

        sensor_data.append((
            current_time,
            loc,
            vehicle
        ))

sensor_df = spark.createDataFrame(sensor_data, ["timestamp", "location", "vehicle_count"])

print("📊 Data berhasil dibuat!")

# =====================================================
# PROCESSING LOGIC
# =====================================================

# 1. Total kendaraan per kota
traffic_df = sensor_df.groupBy("location") \
    .agg(_sum("vehicle_count").alias("total_vehicle")) \
    .orderBy(desc("total_vehicle"))

# 2. Kategori kepadatan (AI logic sederhana)
traffic_df = traffic_df.withColumn(
    "traffic_status",
    when(col("total_vehicle") > 25000, "High 🚨")
    .when(col("total_vehicle") > 15000, "Medium ⚠️")
    .otherwise("Low ✅")
)

# 3. Trend per 10 menit
traffic_time_df = sensor_df.groupBy(
    window(col("timestamp"), "10 minutes"),
    "location"
).agg(_sum("vehicle_count").alias("total_vehicle"))

# 4. Dataset untuk AI / ML
ml_df = sensor_df.withColumn("hour", hour(col("timestamp")))

# =====================================================
# SHOW OUTPUT (BIAR KELIHATAN BAGUS)
# =====================================================

print("\n🔥 TOP TRAFFIC PER KOTA:")
traffic_df.show(10, False)

print("\n📈 TREND TRAFFIC:")
traffic_time_df.show(10, False)

# =====================================================
# SAVE TO PARQUET
# =====================================================

def save_data(df, folder_name):
    path = os.path.join(OUTPUT_DIR, folder_name)
    print(f"💾 Menyimpan ke: {path}")
    df.write.mode("overwrite").parquet(path)

try:
    save_data(traffic_df, "traffic_summary")
    save_data(traffic_time_df, "traffic_trend")
    save_data(ml_df, "ml_ready")

    print("\n✅ SEMUA DATA BERHASIL DISIMPAN!")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")

# =====================================================
# STOP SPARK
# =====================================================

spark.stop()
print("🛑 Spark Session Closed. Silakan jalankan Streamlit sekarang.")