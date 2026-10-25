"""NYC TaxiPulse dashboard (Streamlit). Reads ONLY the small curated tables
produced by Spark; it never touches the 19M-row trip data.

    streamlit run taxipulse/app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taxipulse import config  # noqa: E402

st.set_page_config(page_title="NYC TaxiPulse", page_icon="🚕", layout="wide")
DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


@st.cache_data
def load(name: str) -> pd.DataFrame:
    p = config.CURATED_DIR / f"{name}.parquet"
    if not p.exists():
        st.error(f"Missing {p.name}. Run: python -m taxipulse.pipeline.run_pipeline")
        st.stop()
    return pd.read_parquet(p)


@st.cache_data
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


overview = load_json(config.CURATED_DIR / "overview.json")
if overview is None:
    st.error("No curated data found. Run the Spark pipeline first: python -m taxipulse.pipeline.run_pipeline")
    st.stop()

st.sidebar.title("🚕 NYC TaxiPulse")
st.sidebar.caption(f"Yellow Taxi · {overview['date_from']} → {overview['date_to']} · {overview['total_trips']:,} cleaned trips")
page = st.sidebar.radio("Explore", ["Overview", "Demand", "Zones", "Routes", "Economics", "Prediction", "Data quality & Benchmark"])

# ------------------------------------------------------------- overview --
if page == "Overview":
    st.title("What is happening across NYC taxi activity?")
    c = st.columns(5)
    c[0].metric("Total trips", f"{overview['total_trips']/1e6:.2f} M")
    c[1].metric("Total revenue", f"${overview['total_revenue']/1e6:.1f} M")
    c[2].metric("Avg fare", f"${overview['avg_fare']:.2f}")
    c[3].metric("Avg distance", f"{overview['avg_distance_mi']:.2f} mi")
    c[4].metric("Avg duration", f"{overview['avg_duration_min']:.1f} min")
    daily = load("daily_metrics")
    st.plotly_chart(px.line(daily, x="pickup_date", y="trip_count", title="Daily trips (all zones)", labels={"pickup_date": "Date", "trip_count": "Trips"}), use_container_width=True)
    col1, col2 = st.columns(2)
    hp = load("hourly_profile")
    col1.plotly_chart(px.bar(hp, x="pickup_hour", y="trip_count", title="Trips by hour of day", labels={"pickup_hour": "Hour", "trip_count": "Trips"}), use_container_width=True)
    bm = load("borough_metrics")
    col2.plotly_chart(px.bar(bm, x="borough", y="trip_count", title="Pickups by borough", labels={"trip_count": "Trips"}), use_container_width=True)

# --------------------------------------------------------------- demand --
elif page == "Demand":
    st.title("When is taxi demand highest?")
    hm = load("hour_dow_heatmap")
    piv = hm.pivot(index="dow_name", columns="pickup_hour", values="trip_count").reindex(DOW)
    st.plotly_chart(px.imshow(piv, aspect="auto", title="Demand heatmap: weekday × hour (trips)", labels={"x": "Hour", "y": "Weekday", "color": "Trips"}), use_container_width=True)
    col1, col2 = st.columns(2)
    ww = load("weekend_vs_weekday_hourly")
    ww["day_type"] = ww["is_weekend"].map({True: "Weekend", False: "Weekday"})
    col1.plotly_chart(px.line(ww, x="pickup_hour", y="trip_count", color="day_type", title="Weekday vs weekend hourly demand", labels={"pickup_hour": "Hour", "trip_count": "Trips"}), use_container_width=True)
    mm = load("monthly_metrics")
    col2.plotly_chart(px.bar(mm, x="year_month", y="trip_count", title="Trips by month", labels={"year_month": "Month", "trip_count": "Trips"}), use_container_width=True)
    dp = load("dow_profile")
    st.plotly_chart(px.bar(dp, x="dow_name", y="trip_count", title="Trips by weekday", labels={"dow_name": "Day", "trip_count": "Trips"}), use_container_width=True)

# ---------------------------------------------------------------- zones --
elif page == "Zones":
    st.title("Which taxi zones are busiest?")
    zm = load("zone_metrics")
    boroughs = ["All"] + sorted(zm["borough"].dropna().unique().tolist())
    b = st.selectbox("Borough", boroughs)
    metric = st.selectbox("Metric", ["pickup_trips", "dropoff_trips", "avg_fare", "avg_distance_mi", "avg_duration_min", "avg_tip_pct", "pickup_revenue"])
    sub = zm if b == "All" else zm[zm["borough"] == b]
    top = sub.sort_values(metric, ascending=False).head(20)
    st.plotly_chart(px.bar(top, x=metric, y="zone", orientation="h", color="borough", title=f"Top 20 zones by {metric}").update_layout(yaxis=dict(autorange="reversed")), use_container_width=True)
    zone = st.selectbox("Inspect a zone", sub.sort_values("pickup_trips", ascending=False)["zone"].tolist())
    z = zm[zm["zone"] == zone].iloc[0]
    c = st.columns(5)
    c[0].metric("Pickups", f"{int(z.pickup_trips):,}"); c[1].metric("Drop-offs", f"{int(z.dropoff_trips):,}")
    c[2].metric("Avg fare", f"${z.avg_fare:.2f}"); c[3].metric("Avg distance", f"{z.avg_distance_mi:.2f} mi"); c[4].metric("Avg tip (card)", f"{z.avg_tip_pct:.1f}%")
    zh = load("zone_hourly_profile")
    st.plotly_chart(px.bar(zh[zh["zone_id"] == int(z.zone_id)], x="pickup_hour", y="trip_count", title=f"Hourly pickup profile: {zone}", labels={"pickup_hour": "Hour", "trip_count": "Trips"}), use_container_width=True)

# --------------------------------------------------------------- routes --
elif page == "Routes":
    st.title("Which pickup → drop-off combinations are most common?")
    rm = load("route_metrics")
    rm["route"] = rm["pu_zone"] + " → " + rm["do_zone"]
    n = st.slider("Top N routes", 5, 50, 20)
    st.plotly_chart(px.bar(rm.head(n), x="trip_count", y="route", orientation="h", title=f"Top {n} zone-to-zone routes", hover_data=["avg_fare", "avg_distance_mi", "avg_duration_min"]).update_layout(yaxis=dict(autorange="reversed")), use_container_width=True)
    bf = load("borough_flows")
    piv = bf.pivot(index="pu_borough", columns="do_borough", values="trip_count").fillna(0)
    st.plotly_chart(px.imshow(piv, text_auto=".2s", aspect="auto", title="Borough-to-borough flows (trips)", labels={"x": "Drop-off borough", "y": "Pickup borough"}), use_container_width=True)

# ------------------------------------------------------------ economics --
elif page == "Economics":
    st.title("How do fare, distance, tips and payment behaviour vary?")
    col1, col2 = st.columns(2)
    fd = load("fare_by_distance")
    col1.plotly_chart(px.line(fd, x="distance_bin_mi", y="avg_fare", title="Average fare vs trip distance", labels={"distance_bin_mi": "Distance (mi)", "avg_fare": "Avg fare ($)"}), use_container_width=True)
    pt = load("payment_type_breakdown")
    col2.plotly_chart(px.pie(pt, names="payment_name", values="trip_count", title="Payment method share"), use_container_width=True)
    col3, col4 = st.columns(2)
    th = load("tips_by_hour")
    col3.plotly_chart(px.line(th, x="pickup_hour", y="avg_tip_pct", title="Average tip % (card payments) by hour", labels={"pickup_hour": "Hour", "avg_tip_pct": "Tip %"}), use_container_width=True)
    dd = load("duration_distribution")
    col4.plotly_chart(px.bar(dd, x="duration_bin_min", y="trip_count", title="Trip duration distribution (5-min bins)", labels={"duration_bin_min": "Duration (min)", "trip_count": "Trips"}), use_container_width=True)
    hp = load("hourly_profile")
    st.plotly_chart(px.line(hp, x="pickup_hour", y="avg_fare", title="Average fare by hour of day", labels={"pickup_hour": "Hour", "avg_fare": "Avg fare ($)"}), use_container_width=True)

# ----------------------------------------------------------- prediction --
elif page == "Prediction":
    st.title("Predicted taxi demand for a zone and hour")
    st.info("These are **model outputs**, not observations. See expected error below.")
    try:
        from taxipulse.ml import predict
        zones_df = load("zone_metrics").set_index("zone_id")
        zids = predict.available_zones()
        zid = st.selectbox("Zone", zids, format_func=lambda i: f"{zones_df.loc[i, 'zone']} ({zones_df.loc[i, 'borough']})" if i in zones_df.index else str(i), index=zids.index(161) if 161 in zids else 0)
        col1, col2 = st.columns(2)
        d = col1.date_input("Date", pd.Timestamp("2026-06-05").date())
        h = col2.slider("Hour", 0, 23, 18)
        res = predict.predict_calendar(zid, pd.Timestamp(d) + pd.Timedelta(hours=h))
        c = st.columns(3)
        c[0].metric("Predicted demand", f"~{res['predicted_trips']:.0f} trips/hr")
        c[1].metric("Historical mean (same hour & weekday)", f"{res['historical_mean_same_hour_dow']:.0f} trips/hr")
        c[2].metric("Expected error (MAE on May 2026)", f"± {res['expected_error_mae']:.1f} trips")
        st.caption("Calendar-only model: usable for any future hour. The lag model below is more accurate but needs the most recent hours of real data.")
        hist = predict.recent_history(zid, 72)
        fut = predict.predict_next_hours(zid, 12)
        plot = pd.concat([hist.rename(columns={"actual_trips": "trips"}).assign(kind="actual"), fut.rename(columns={"predicted_trips": "trips"}).assign(kind="predicted")])
        st.plotly_chart(px.line(plot, x="timestamp", y="trips", color="kind", line_dash="kind", title="Last 72h actual demand + next 12h forecast (lag model)"), use_container_width=True)
        m = predict.model_metrics()
        st.subheader("Model evaluation (chronological holdout: train January to April, test May 2026)")
        st.dataframe(pd.DataFrame(m["models"]).T[["MAE", "RMSE", "R2"]])
        st.caption(f"Lag model improves MAE by {m['improvement_over_last_week_baseline_pct']}% over the same-hour-last-week baseline.")
        fi = load_json(config.MODELS_DIR / "feature_importance.json")
        if fi:
            st.plotly_chart(px.bar(pd.DataFrame(fi["lightgbm_with_lags"], columns=["feature", "importance"]), x="importance", y="feature", orientation="h", title="Feature importance (lag model)").update_layout(yaxis=dict(autorange="reversed")), use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Prediction unavailable: {e}")

# -------------------------------------------------- quality + benchmark --
else:
    st.title("What happened to the data, and why Spark?")
    q = load_json(config.REPORTS_DIR / "quality_report.json")
    if q:
        c = st.columns(4)
        c[0].metric("Raw records", f"{q['raw_records']:,}"); c[1].metric("Retained", f"{q['retained_records']:,}")
        c[2].metric("Removed", f"{q['removed_records']:,}"); c[3].metric("Retention", f"{100*q['retention_rate']:.2f}%")
        rr = pd.DataFrame(list(q["removed_by_reason"].items()), columns=["reason", "records"])
        st.plotly_chart(px.bar(rr, x="records", y="reason", orientation="h", title="Records removed by rule (first violated rule)").update_layout(yaxis=dict(autorange="reversed")), use_container_width=True)
        st.write("Null rate in raw data (fraction):", pd.Series(q["null_rate_raw"]).sort_values(ascending=False).head(6))
    b = load_json(config.REPORTS_DIR / "benchmark_results.json")
    if b:
        st.subheader("Spark performance experiments (measured)")
        st.caption(f"Environment: {b['environment']['cpu_cores']} cores · {b['environment']['ram_gb']} GB RAM · PySpark {b['environment']['pyspark']} · {b['environment']['dataset_rows']:,} rows · {b['environment']['dataset_size_mb']} MB Parquet · query: {b['query']}")
        st.dataframe(pd.DataFrame(b["results"]).drop(columns=["runs_seconds"]))
    else:
        st.info("Run: python -m taxipulse.benchmark.run_benchmark")
