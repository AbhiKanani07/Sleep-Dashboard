import os
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
try:
    import shap
except Exception:
    shap = None
import streamlit as st

st.set_page_config(page_title="Sleep Insights Dashboard", layout="wide")

# --------------------------------------------------
# Data loading and prep
# --------------------------------------------------
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Fitbit and Apple data.
    If env vars FITBIT_CSV_URL / APPLE_CSV_URL are set, load from those URLs (useful for hosted deploys).
    Otherwise, load local CSVs from data/clean/ relative to this file.
    """
    app_dir = Path(__file__).resolve().parent
    data_dir = app_dir / "data" / "clean"

    def resolve_path(env_var: str, default_filename: str) -> str:
        val = os.getenv(env_var)
        if val:
            return val  # allow URL or custom path
        # local default paths (current dir and parent dir fallbacks)
        local_path = data_dir / default_filename
        if local_path.exists():
            return local_path.as_posix()
        parent_path = (app_dir.parent / "data" / "clean" / default_filename)
        if parent_path.exists():
            return parent_path.as_posix()
        return local_path.as_posix()  # return default even if missing for error message

    fitbit_src = resolve_path("FITBIT_CSV_URL", "fitbit_clean.csv")
    apple_src = resolve_path("APPLE_CSV_URL", "apple_sleep_nightly_summary.csv")

    try:
        fitbit = pd.read_csv(fitbit_src, parse_dates=["DATE"])
        apple = pd.read_csv(apple_src, parse_dates=["sleep_date"])
    except FileNotFoundError:
        st.error(
            "Data files not found. Ensure these files exist in your repository:\n"
            f"- {fitbit_src}\n"
            f"- {apple_src}\n"
            "Or set environment variables FITBIT_CSV_URL and APPLE_CSV_URL to point to your data."
        )
        st.stop()

    apple["DATE"] = apple["sleep_date"]
    return fitbit, apple


def enrich_fitbit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bedtime_num"] = pd.to_datetime(df["BEDTIME"]).dt.hour + pd.to_datetime(df["BEDTIME"]).dt.minute / 60
    df["wakeup_num"] = pd.to_datetime(df["WAKEUP"]).dt.hour + pd.to_datetime(df["WAKEUP"]).dt.minute / 60
    df["weekday"] = df["DATE"].dt.day_name()
    df["week"] = df["DATE"].dt.isocalendar().week
    df["weekday_num"] = df["DATE"].dt.weekday
    # midsleep (approx)
    def midsleep(row):
        bt = row["bedtime_num"]
        wu = row["wakeup_num"]
        if pd.isna(bt) or pd.isna(wu):
            return np.nan
        if wu < bt:
            wu += 24
        mid = (bt + wu) / 2.0
        return mid if mid < 24 else mid - 24
    df["midsleep"] = df.apply(midsleep, axis=1)
    return df


def apply_filters(
    df: pd.DataFrame,
    date_col: str,
    date_range: List[pd.Timestamp],
    weekdays: List[str],
) -> pd.DataFrame:
    start, end = date_range
    mask = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
    if weekdays:
        mask &= df[date_col].dt.day_name().isin(weekdays)
    return df.loc[mask].copy()


fitbit_raw, apple_raw = load_data()
fitbit = enrich_fitbit(fitbit_raw)
apple = apple_raw.copy()

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Insights Summary",
        "Sleep Score Predictor",
        "Fitbit Insights",
        "Apple Health Insights",
        "Fitbit vs Apple (No Overlap View)",
        "Weekly & Pattern Analysis",
        "Sleep Regularity & Chronotype",
        "Recommendations",
        "Correlation Analysis",
        "Model Explainability",
    ],
)

st.sidebar.subheader("Filters")
default_range = [fitbit["DATE"].min().date(), fitbit["DATE"].max().date()]
date_range = st.sidebar.date_input(
    "Date range (Fitbit)",
    default_range,
    min_value=default_range[0],
    max_value=default_range[1],
)
if not isinstance(date_range, list) or len(date_range) != 2:
    date_range = default_range
weekdays = st.sidebar.multiselect(
    "Weekdays",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)
grouping = st.sidebar.selectbox("Group by", ["Daily", "Weekly", "Monthly"])

# Apply filters
fitbit_filt = apply_filters(fitbit, "DATE", date_range, weekdays)
apple_filt = apply_filters(apple, "DATE", date_range, weekdays)
# If the shared date filter wipes out Apple data (no overlap), fall back to full Apple dataset
if apple_filt.empty:
    apple_filt = apple.copy()


def group_df(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    if grouping == "Weekly":
        df["period"] = df[date_col].dt.to_period("W").apply(lambda r: r.start_time)
    elif grouping == "Monthly":
        df["period"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    else:
        df["period"] = df[date_col]
    return df

# --------------------------------------------------
# Helper functions & visuals
# --------------------------------------------------
def line_plot(df: pd.DataFrame, x: str, y: str, title: str, color: str = None, hover: List[str] = None):
    fig = px.line(df, x=x, y=y, color=color, hover_data=hover)
    fig.update_traces(mode="lines+markers")
    fig.update_layout(title=title, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def dist_plot(fit_series, apple_series, title: str, xaxis: str):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=fit_series, name="Fitbit", opacity=0.6))
    fig.add_trace(go.Histogram(x=apple_series, name="Apple", opacity=0.6))
    fig.update_layout(barmode="overlay", title=title, xaxis_title=xaxis)
    fig.update_traces(hovertemplate="%{x:.2f}")
    st.plotly_chart(fig, use_container_width=True)


def download_button(df: pd.DataFrame, label: str, file_name: str):
    csv = df.to_csv(index=False)
    st.download_button(label=label, data=csv, file_name=file_name, mime="text/csv")


def safe_mean(series: pd.Series, decimals: int = 2):
    s = series.dropna()
    if s.empty:
        return None
    return round(s.mean(), decimals)


def fmt_value(value, decimals: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def fmt_range(series: pd.Series, decimals: int = 1, suffix: str = "") -> str:
    s = series.dropna()
    if s.empty:
        return "No data"
    return f"{s.min():.{decimals}f}{suffix} to {s.max():.{decimals}f}{suffix}"

# ---------- Sleep regularity & chronotype helpers ----------

def compute_regularities(df: pd.DataFrame):
    out = {}
    if df.empty:
        return out

    out["avg_sleep_hours"] = safe_mean(df["HOURS_OF_SLEEP_HOURS"])
    out["avg_deep_pct"] = safe_mean(df.get("DEEP_SLEEP", pd.Series(dtype=float)), 1)

    # Bedtime std
    out["bedtime_std"] = df["bedtime_num"].std()

    # Social jetlag: weekend vs weekday midsleep
    weekend = df[df["weekday"].isin(["Saturday", "Sunday"])]
    weekday = df[~df["weekday"].isin(["Saturday", "Sunday"])]

    def avg_midsleep(sub):
        s = sub["midsleep"].dropna()
        return s.mean() if not s.empty else np.nan

    weekday_mid = avg_midsleep(weekday)
    weekend_mid = avg_midsleep(weekend)
    out["weekday_mid"] = weekday_mid
    out["weekend_mid"] = weekend_mid

    if not np.isnan(weekday_mid) and not np.isnan(weekend_mid):
        out["social_jetlag"] = abs(weekend_mid - weekday_mid)
    else:
        out["social_jetlag"] = np.nan

    # Approx Sleep Regularity Index (pseudo): based on bedtime std scaled 0-100
    if pd.isna(out["bedtime_std"]):
        out["sri"] = np.nan
    else:
        val = 1 - min(out["bedtime_std"], 3) / 3  # <=0.5h is excellent, >=3h is poor
        out["sri"] = max(0, val) * 100

    # Chronotype: based on weekend midsleep (if available), otherwise overall
    ms = weekend_mid if not np.isnan(weekend_mid) else avg_midsleep(df)
    out["midsleep_chrono"] = ms
    if pd.isna(ms):
        out["chronotype"] = "Unknown"
    else:
        if ms < 3:
            out["chronotype"] = "Extreme Morning"
        elif ms < 5:
            out["chronotype"] = "Morning"
        elif ms < 7:
            out["chronotype"] = "Intermediate"
        elif ms < 9:
            out["chronotype"] = "Evening"
        else:
            out["chronotype"] = "Extreme Evening"

    return out


def generate_insights(fit_df: pd.DataFrame, apple_df: pd.DataFrame) -> List[str]:
    insights = []
    if fit_df.empty and apple_df.empty:
        return ["No data available for the selected filters."]

    # weekend vs weekday sleep duration
    if not fit_df.empty:
        weekend = fit_df[fit_df["weekday"].isin(["Saturday", "Sunday"])]
        weekday = fit_df[~fit_df["weekday"].isin(["Saturday", "Sunday"])]
        if not weekend.empty and not weekday.empty:
            wkd = weekend["HOURS_OF_SLEEP_HOURS"].mean()
            wd = weekday["HOURS_OF_SLEEP_HOURS"].mean()
            diff = wkd - wd
            if abs(diff) >= 0.5:
                direction = "longer" if diff > 0 else "shorter"
                insights.append(
                    f"You sleep about {abs(diff):.1f} hours {direction} on weekends compared to weekdays (Fitbit)."
                )

        # deep sleep relationship with score
        if "DEEP_SLEEP" in fit_df.columns and "SLEEP_SCORE" in fit_df.columns:
            corr = fit_df["DEEP_SLEEP"].corr(fit_df["SLEEP_SCORE"])
            if not pd.isna(corr):
                if corr > 0.3:
                    insights.append("Nights with higher deep sleep % tend to have higher sleep scores (Fitbit).")
                elif corr < -0.3:
                    insights.append("Nights with higher deep sleep % surprisingly correlate with lower sleep scores (Fitbit).")

        # bedtime variability
        bt_std = fit_df["bedtime_num"].std()
        if not pd.isna(bt_std):
            if bt_std < 0.75:
                insights.append(f"Your bedtime is very consistent (std dev ~ {bt_std:.2f} hours).")
            elif bt_std > 1.5:
                insights.append(f"Your bedtime varies quite a bit (std dev ~ {bt_std:.2f} hours).")

    if not apple_df.empty:
        if "pct_deep" in apple_df.columns:
            avg_deep = apple_df["pct_deep"].mean()
            insights.append(f"Your Apple Watch reports average deep sleep of about {avg_deep:.1f}%.")
    if not insights:
        insights.append("No strong patterns detected; try widening the date range or including more weekdays.")
    return insights


def generate_recommendations(metrics: dict) -> List[str]:
    recs = []
    if not metrics:
        return ["No data available to generate recommendations."]

    avg_sleep = metrics.get("avg_sleep_hours")
    deep_pct = metrics.get("avg_deep_pct")
    sri = metrics.get("sri")
    sj = metrics.get("social_jetlag")
    chrono = metrics.get("chronotype")

    if avg_sleep is not None:
        if avg_sleep < 7:
            recs.append(
                f"- You're averaging {avg_sleep:.1f} hours of sleep. Aim for at least 7-8 hours by prioritizing a consistent wind-down routine."
            )
        else:
            recs.append(
                f"- Your average sleep duration is {avg_sleep:.1f} hours, which is within a healthy range. Focus on quality and consistency."
            )

    if deep_pct is not None:
        if deep_pct < 15:
            recs.append(
                f"- Deep sleep averages {deep_pct:.1f}%. Consider reducing late caffeine, heavy meals, and screens close to bedtime."
            )
        elif deep_pct > 25:
            recs.append(
                f"- Deep sleep averages {deep_pct:.1f}%, which is quite robust. Maintain habits that support this (consistent schedule, good sleep environment)."
            )

    if sri is not None and not pd.isna(sri):
        if sri < 60:
            recs.append(
                f"- Your sleep regularity score (~{sri:.0f}/100) suggests varying bed/wake times. Try keeping bedtime within a 30-45 minute window each day."
            )
        else:
            recs.append(
                f"- Your sleep regularity score (~{sri:.0f}/100) indicates reasonably consistent bed/wake times. This is great for circadian health."
            )

    if sj is not None and not pd.isna(sj):
        if sj > 2:
            recs.append(
                f"- Social jetlag (~{sj:.1f} hours difference between weekday/weekend midsleep) is high. Try aligning weekend schedules closer to weekdays."
            )

    if chrono and chrono != "Unknown":
        recs.append(
            f"- Your chronotype appears to be **{chrono}**. When possible, schedule demanding tasks to match your natural energy peaks."
        )

    if not recs:
        recs.append("No specific recommendations found. Try expanding your date range or ensuring enough valid nights.")
    return recs

# --------------------------------------------------
# Pages
# --------------------------------------------------
if page == "Insights Summary":
    st.title("Insights Summary")
    apple_view = apple_filt if not apple_filt.empty else apple
    fitbit_view = fitbit_filt if not fitbit_filt.empty else fitbit

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fitbit avg sleep (hrs)", fmt_value(safe_mean(fitbit_view["HOURS_OF_SLEEP_HOURS"]), 2))
    col2.metric("Fitbit avg score", fmt_value(safe_mean(fitbit_view["SLEEP_SCORE"], 1), 1))
    col3.metric("Apple avg sleep (hrs)", fmt_value(safe_mean(apple_view["total_sleep_hours"]), 2))
    col4.metric("Apple avg deep %", fmt_value(safe_mean(apple_view["pct_deep"], 1), 1))

    st.subheader("Key highlights")
    st.write(
        "- Fitbit sleep hours range: "
        f"{fmt_range(fitbit_view['HOURS_OF_SLEEP_HOURS'], 2, ' hrs')}"
    )
    st.write(
        "- Apple deep sleep range: "
        f"{fmt_range(apple_view['pct_deep'], 1, '%')}"
    )

    # NEW: Insights engine
    st.subheader("Auto-generated insights")
    insights = generate_insights(fitbit_view, apple_view)
    for text in insights:
        st.write(f"- {text}")

    with st.expander("Download filtered data"):
        download_button(fitbit_filt, "Download Fitbit CSV", "fitbit_filtered.csv")
        download_button(apple_filt, "Download Apple CSV", "apple_filtered.csv")
    st.caption("Data sources: local CSVs under data/clean/, override via FITBIT_CSV_URL and APPLE_CSV_URL (useful on Hugging Face Spaces).")

elif page == "Sleep Score Predictor":
    st.title("Sleep Score Predictor")
    MODEL_PATH = Path("models/sleep_rf_model.pkl")
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    if model is None:
        st.warning("Model not found. Place your model at models/sleep_rf_model.pkl")
    else:
        st.caption("Sleep Score is Fitbit's 0-100 estimate of overnight recovery and sleep quality based on duration, depth, and heart metrics.")
        st.markdown("### Baseline Inputs")
        col1, col2 = st.columns(2)
        with col1:
            hours = st.slider("Hours slept", 0, 12, 7)
            rem = st.slider("REM %", 0, 50, 20)
        with col2:
            deep = st.slider("Deep %", 0, 50, 15)
            hr = st.slider("HR Below Resting %", 0, 100, 90)

        baseline_df = pd.DataFrame(
            [[hours, rem, deep, hr]],
            columns=[
                "HOURS_OF_SLEEP_HOURS",
                "REM_SLEEP",
                "DEEP_SLEEP",
                "HEART_RATE_UNDER_RESTING",
            ],
        )
        baseline_pred = model.predict(baseline_df)[0]

        st.metric("Predicted Sleep Score (baseline)", f"{baseline_pred:.1f}")

        st.markdown("### What-if Scenario (simulate changes)")
        col3, col4 = st.columns(2)
        with col3:
            hours2 = st.slider("What-if: Hours slept", 0, 12, hours)
            rem2 = st.slider("What-if: REM %", 0, 50, rem)
        with col4:
            deep2 = st.slider("What-if: Deep %", 0, 50, deep)
            hr2 = st.slider("What-if: HR Below Resting %", 0, 100, hr)

        scenario_df = pd.DataFrame(
            [[hours2, rem2, deep2, hr2]],
            columns=[
                "HOURS_OF_SLEEP_HOURS",
                "REM_SLEEP",
                "DEEP_SLEEP",
                "HEART_RATE_UNDER_RESTING",
            ],
        )
        scenario_pred = model.predict(scenario_df)[0]
        delta = scenario_pred - baseline_pred

        col_a, col_b = st.columns(2)
        col_a.metric("What-if Sleep Score", f"{scenario_pred:.1f}", f"{delta:+.1f} vs baseline")
        col_b.write(
            "Try increasing hours or deep sleep to see how much the model thinks your score can improve."
        )

elif page == "Fitbit Insights":
    st.title("Fitbit Insights")
    g_fit = group_df(fitbit_filt, "DATE")

    st.subheader("Sleep Duration")
    line_plot(g_fit, "period", "HOURS_OF_SLEEP_HOURS", "Fitbit Sleep Duration", hover=["SLEEP_SCORE"])

    st.subheader("Sleep Score")
    line_plot(g_fit, "period", "SLEEP_SCORE", "Fitbit Sleep Score")

    st.subheader("Bedtime vs Wakeup")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g_fit["period"], y=g_fit["bedtime_num"], mode="lines+markers", name="Bedtime"))
    fig.add_trace(go.Scatter(x=g_fit["period"], y=g_fit["wakeup_num"], mode="lines+markers", name="Wakeup"))
    fig.update_layout(title="Bedtime and Wakeup", yaxis_title="Hour of day", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Download")
    download_button(g_fit, "Download Fitbit (filtered)", "fitbit_filtered.csv")

elif page == "Apple Health Insights":
    st.title("Apple Health Insights")
    g_apple = group_df(apple_filt, "DATE")

    st.subheader("Deep Sleep %")
    line_plot(g_apple, "period", "pct_deep", "Apple Deep Sleep %")

    st.subheader("Sleep Duration")
    line_plot(g_apple, "period", "total_sleep_hours", "Apple Sleep Duration (hrs)")

    st.subheader("Heart Rate During Sleep")
    line_plot(g_apple, "period", "avg_sleep_hr", "Avg Sleep HR (bpm)")

    st.subheader("Download")
    download_button(g_apple, "Download Apple (filtered)", "apple_filtered.csv")

elif page == "Fitbit vs Apple (No Overlap View)":
    st.title("Fitbit vs Apple (No Overlap)")
    apple_view = apple_filt if not apple_filt.empty else apple
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fitbit Summary")
        st.metric("Avg sleep (hrs)", fmt_value(safe_mean(fitbit_filt["HOURS_OF_SLEEP_HOURS"]), 2))
        st.metric("Avg deep %", fmt_value(safe_mean(fitbit_filt["DEEP_SLEEP"], 1), 1))
        line_plot(fitbit_filt, "DATE", "HOURS_OF_SLEEP_HOURS", "Fitbit Sleep Hours")
    with col2:
        st.subheader("Apple Summary")
        if apple_view.empty:
            st.info("No Apple data to display for the current filters.")
        else:
            st.metric("Avg sleep (hrs)", fmt_value(safe_mean(apple_view["total_sleep_hours"]), 2))
            st.metric("Avg deep %", fmt_value(safe_mean(apple_view["pct_deep"], 1), 1))
            line_plot(apple_view, "DATE", "total_sleep_hours", "Apple Sleep Hours")

    st.subheader("Sleep Hours Distribution")
    if apple_view.empty:
        st.info("No Apple data for distribution charts with current filters.")
    else:
        dist_plot(fitbit_filt["HOURS_OF_SLEEP_HOURS"], apple_view["total_sleep_hours"], "Sleep Hours Distribution", "Hours")

    st.subheader("Deep Sleep % Distribution")
    if apple_view.empty:
        st.info("No Apple data for distribution charts with current filters.")
    else:
        dist_plot(fitbit_filt["DEEP_SLEEP"], apple_view["pct_deep"], "Deep Sleep % Distribution", "Deep %")

elif page == "Weekly & Pattern Analysis":
    st.title("Weekly & Pattern Analysis (Fitbit)")
    st.subheader("Average Sleep by Weekday")
    avg_sleep = fitbit_filt.groupby("weekday")["HOURS_OF_SLEEP_HOURS"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    fig = px.bar(avg_sleep, labels={"value": "Hours"}, title="Average Sleep by Weekday")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sleep Duration Heatmap (Week x Day)")
    pivot = fitbit_filt.pivot_table(index="week", columns="weekday_num", values="HOURS_OF_SLEEP_HOURS")
    if pivot.empty:
        st.info("Not enough data to render a heatmap for the selected filters.")
    else:
        heatmap_fig = px.imshow(pivot, labels={"color": "Hours"}, x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        st.plotly_chart(heatmap_fig, use_container_width=True)

elif page == "Sleep Regularity & Chronotype":
    st.title("Sleep Regularity & Chronotype")

    metrics = compute_regularities(fitbit_filt if not fitbit_filt.empty else fitbit)
    if not metrics:
        st.info("Not enough data to compute regularity metrics.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Sleep Regularity Index (approx)", fmt_value(metrics["sri"], 0, "/100"))
        col2.metric("Bedtime std dev (hrs)", fmt_value(metrics["bedtime_std"], 2))
        col3.metric("Social jetlag (hrs)", fmt_value(metrics["social_jetlag"], 2))

        st.subheader("Chronotype")
        st.write(f"Estimated chronotype: **{metrics['chronotype']}**")
        if metrics["midsleep_chrono"] is not None and not pd.isna(metrics["midsleep_chrono"]):
            st.write(f"Average midsleep time: ~{metrics['midsleep_chrono']:.1f} hours after midnight.")

        st.caption("Note: SRI and chronotype here are approximations based on simplified formulas, not clinical diagnostics.")

elif page == "Recommendations":
    st.title("Personalized Recommendations")
    metrics = compute_regularities(fitbit_filt if not fitbit_filt.empty else fitbit)
    recs = generate_recommendations(metrics)
    st.markdown("### Suggestions based on your current patterns:")
    for r in recs:
        st.write(r)

elif page == "Correlation Analysis":
    st.title("Correlation Analysis")
    st.subheader("Heart Rate vs Sleep Duration (Fitbit)")
    corr_df = fitbit_filt.dropna(subset=["HEART_RATE_UNDER_RESTING", "HOURS_OF_SLEEP_HOURS"])
    if len(corr_df) < 2:
        st.info("Not enough data for a trend line.")
    else:
        x = corr_df["HEART_RATE_UNDER_RESTING"].astype(float)
        y = corr_df["HOURS_OF_SLEEP_HOURS"].astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 2)
        line_y = slope * line_x + intercept

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name="Data",
            hovertemplate="HR Under Resting: %{x:.2f}<br>Sleep Hours: %{y:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=line_x, y=line_y, mode="lines", name="Trend",
            hovertemplate="Trend<br>HR: %{x:.2f}<br>Sleep Hours: %{y:.2f}<extra></extra>"
        ))
        fig.update_layout(
            title="HR Under Resting vs Sleep Hours",
            xaxis_title="HR Under Resting (%)",
            yaxis_title="Sleep Hours",
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bedtime Variability")
    bedtime_std = fitbit_filt["bedtime_num"].std()
    st.metric("Std Dev of Bedtime (hrs)", f"{bedtime_std:.2f}" if not pd.isna(bedtime_std) else "N/A")

elif page == "Model Explainability":
    st.title("Model Explainability (SHAP)")
    MODEL_PATH = Path("models/sleep_rf_model.pkl")
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    if shap is None:
        st.warning("SHAP is not installed. Install `shap` to view explainability plots.")
    elif model is None:
        st.warning("Model not found. Place your model at models/sleep_rf_model.pkl")
    else:
        st.write("This page shows which features drive the sleep score predictions.")

        # Take a sample of recent nights for SHAP
        sample = fitbit_filt[[
            "HOURS_OF_SLEEP_HOURS",
            "REM_SLEEP",
            "DEEP_SLEEP",
            "HEART_RATE_UNDER_RESTING",
        ]].dropna().tail(100)

        if sample.empty:
            st.info("Not enough data for SHAP analysis with current filters.")
        else:
            try:
                explainer = shap.Explainer(model)
                shap_values = explainer(sample)

                st.subheader("Feature importance (global)")
                shap.plots.bar(shap_values, show=False)
                fig_bar = plt.gcf()
                st.pyplot(fig_bar, clear_figure=True)
                plt.clf()

                st.subheader("Per-night explanation (beeswarm)")
                shap.plots.beeswarm(shap_values, show=False)
                fig_bee = plt.gcf()
                st.pyplot(fig_bee, clear_figure=True)
                plt.clf()

                st.caption("Note: SHAP plots approximate how each feature pushed predictions up or down.")
            except Exception as e:
                st.error(f"Failed to compute SHAP values: {e}")
                st.info("Check that your model is a compatible scikit-learn estimator.")
# Footer
st.write("---")
st.caption("Dashboard powered by Fitbit + Apple Health datasets.")


