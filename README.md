# Sleep Data Analysis & Wearable Device Comparison

This project analyzes sleep data collected from **Fitbit**, **Apple Health**, and an external **wearable performance dataset**.  
The goal is to build an end-to-end pipeline that:

- Combines self-tracked sleep logs  
- Cleans and standardizes data from multiple devices  
- Produces nightly summaries  
- Enables analysis of sleep duration, sleep architecture, and heart-rate patterns  
- Compares wearable device performance  
- Serves as a foundation for modeling or dashboards  

---

## Data Sources

### **1. Fitbit Sleep Logs**
Monthly CSVs (Jan–Apr, Nov, Dec 2022) containing:
- Sleep score  
- Hours slept  
- REM %  
- Deep sleep %  
- Heart rate below resting  
- Sleep window (“bedtime – wake time”)

### **2. Apple Health Sleep + Heart Rate**
- `sleep_data.csv` — segment-level sleep staging  
- `heart_rate_data.csv` — continuous HR samples  

### **3. Wearable Device Performance Dataset**
Contains:
- Device name/category  
- Battery life  
- HR accuracy  
- GPS accuracy  
- Sensor count  
- Performance score  

---

## Data Cleaning

### **Fitbit Cleaning Steps**
1. Loaded all monthly CSVs and concatenated them.  
2. Renamed inconsistent first column (e.g., “MARCH”, “APRIL”) → `DAY_OF_WEEK`.  
3. Standardized all column names.  
4. Dropped invalid columns (`SLEEP_SQORE`, `FEBEUARY`).  
5. Unified heart-rate fields under `HEART_RATE_UNDER_RESTING`.  
6. Parsed `DATE` into datetime.  
7. Converted `"H:MM:SS"` → numeric hours (`HOURS_OF_SLEEP_HOURS`).  
8. Converted `%` strings → floats.  
9. Split sleep window into `BEDTIME` and `WAKEUP`.

**Output:**  
- `fitbit_clean.csv`

---

### **Apple Health Cleaning Steps**
1. Parsed timestamps (`Start Time`, `End Time`, `Timestamp`).  
2. Calculated segment duration.  
3. Normalized stage labels (`Light/Core` → `LIGHT`).  
4. Assigned each segment to a `sleep_date`.  
5. Aggregated nightly totals:
   - total sleep hours  
   - deep/light sleep hours  
   - average sleep HR  
   - bedtime/wakeup  
6. Added `% deep` and `% light`.

**Outputs:**  
- `apple_sleep_segments_clean.csv`  
- `apple_sleep_nightly_summary.csv`

---

### **Wearable Device Performance Dataset**
- Clean and ready to use as-is.
- Useful for benchmarking Apple Watch vs Fitbit.

**Output:**  
- `wearable_health_devices_performance_upto_26june2025.csv`

---

## Cleaned File Downloads

### **Fitbit**
- [`fitbit_clean.csv`](sandbox:/mnt/data/fitbit_clean.csv)

### **Apple Health**
- [`apple_sleep_segments_clean.csv`](sandbox:/mnt/data/apple_sleep_segments_clean.csv)  
- [`apple_sleep_nightly_summary.csv`](sandbox:/mnt/data/apple_sleep_nightly_summary.csv)

### **Wearable Device Performance**
- [`wearable_health_devices_performance_upto_26june2025.csv`](sandbox:/mnt/data/wearable_health_devices_performance_upto_26june2025.csv)

---

## Next Steps

### **Planned Project Extensions**
- Merge Apple + Fitbit nightly summaries  
- Create visualizations (sleep duration, bedtime patterns, HR trends)  
- Build predictive models (sleep score, deep sleep %)  
- Develop a Streamlit or Dash dashboard  
- Compare wearable device accuracy and consistency  

---

## **Project Purpose**
- Data engineering & cleaning  
- Pandas preprocessing pipelines  
- Multi-device health data integration  
- Feature engineering for sleep analysis  
- Real-world data science workflow  


