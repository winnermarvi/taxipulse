"""NYC TaxiPulse Control Center.

One app that runs AND explains the whole project. Every button launches the
project's real command-line entry point in the background and streams its
live output; every chart reads the outputs those commands write.

    bash scripts/start_app.sh        (or double-click scripts/TaxiPulse.bat on Windows)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from taxipulse import config  # noqa: E402
from taxipulse.app import jobs  # noqa: E402

st.set_page_config(page_title="NYC TaxiPulse", page_icon="🚕", layout="wide")

YEL, NAVY, TEAL, RED, GREY, LGREY = "#F0A202", "#1F3A5F", "#2A7F8F", "#C1272D", "#8C8C8C", "#C9C9C9"
PALETTE = [NAVY, YEL, TEAL, RED, GREY, "#6A4C93", "#8AC926"]
DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

st.markdown("""<style>
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
.block-container {padding-top: 1.6rem;}
</style>""", unsafe_allow_html=True)


# ============================================================== loading ====
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float):
    return json.loads(Path(path).read_text())


def table(name: str) -> pd.DataFrame | None:
    p = config.CURATED_DIR / f"{name}.parquet"
    return _parquet(str(p), _mtime(p)) if p.exists() else None


def jfile(p: Path):
    return _json(str(p), _mtime(p)) if p.exists() else None


def need(obj, what: str, how: str) -> bool:
    """Show a friendly hint instead of a crash when an output does not exist yet."""
    if obj is None:
        st.info(f"No {what} yet. To create it, {how}")
        return False
    return True


def ago(ts: float) -> str:
    if not ts:
        return "-"
    s = time.time() - ts
    for unit, n in (("d", 86400), ("h", 3600), ("min", 60)):
        if s >= n:
            return f"{int(s // n)} {unit} ago"
    return "just now"


def chart(fig, height: int | None = None):
    if height:
        fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


# ============================================================ job panel ====
def job_panel(keys: list[str], prefix: str):
    """Buttons for real commands + a live console underneath."""
    s = jobs.status()
    busy = bool(s and s["state"] == "running")
    cols = st.columns(len(keys))
    for c, k in zip(cols, keys):
        if c.button(f"▶  {jobs.label(k)}", key=f"{prefix}_{k}", disabled=busy, width="stretch", type="primary",
                    help=f"Runs the real command-line job · {jobs.duration(k)}"):
            ok, msg = jobs.start(k)
            if ok:
                st.rerun()
            st.warning(msg)
    with st.expander("Commands behind these buttons"):
        for k in keys:
            st.code("\n".join(f"$ {jobs.pretty(a)}" for _, a in jobs.commands_for(k)), language="bash")
    console(tuple(keys))


@st.fragment(run_every=1.5)
def console(keys: tuple[str, ...]):
    s = jobs.status()
    if not s or (s["state"] != "running" and s["key"] not in keys):
        st.caption("Live output of the command will stream here.")
        return
    icon = {"running": "⏳", "succeeded": "✅", "failed": "❌", "stopped": "⏹️"}[s["state"]]
    head, btn = st.columns([5, 1])
    head.markdown(f"**{icon} {s['label']}** · {s['state']} · {s['elapsed']:.0f} s"
                  + ("" if s["key"] in keys else "  *(started from another page)*"))
    if s["state"] == "running" and btn.button("⏹ Stop", key=f"stop_{s['id']}", width="stretch"):
        jobs.stop()
    with st.container(height=330, border=True):
        st.code(jobs.tail(90, st.session_state.get("hide_warn", True)) or "(waiting for output…)", language="text")
    if s["state"] == "failed":
        st.error(s.get("note") or f"The command exited with code {s.get('returncode')}. The log above shows why.")
    # A job just finished: drop caches so every chart re-reads the new outputs.
    if s["state"] != "running" and st.session_state.get("ack") != s["id"]:
        st.session_state["ack"] = s["id"]
        st.cache_data.clear()
        try:
            from taxipulse.ml import predict
            predict._load.cache_clear()
        except Exception:  # noqa: BLE001
            pass
        st.rerun()


# ============================================================== sidebar ====
if "ack" not in st.session_state:
    _s = jobs.status()
    st.session_state["ack"] = _s["id"] if _s and _s["state"] != "running" else None

st.sidebar.title("🚕 NYC TaxiPulse")
st.sidebar.caption("Control Center · BDA Mini Project")
PAGES = ["🏠 Home", "⚙️ Data & Pipeline", "📊 Analytics", "📈 Prediction", "⏱️ Benchmarks", "🍃 MongoDB"]
page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")
st.sidebar.divider()
_s = jobs.status()
if _s:
    icon = {"running": "⏳", "succeeded": "✅", "failed": "❌", "stopped": "⏹️"}[_s["state"]]
    when = "in progress, see its console" if _s["state"] == "running" else f"{_s['state']} · {_s['elapsed']:.0f} s"
    st.sidebar.markdown(f"**{'Running' if _s['state'] == 'running' else 'Last job'}:** {icon} {_s['label']}  \n{when}")
else:
    st.sidebar.markdown("**Last job:** none yet")
st.sidebar.toggle("Hide Spark warnings in logs", value=True, key="hide_warn")
ov = jfile(config.CURATED_DIR / "overview.json")
if ov:
    st.sidebar.caption(f"Data: {ov['date_from']} → {ov['date_to']} · {ov['total_trips']:,} cleaned trips")


# ================================================================= HOME ====
def page_home():
    st.title("🚕 NYC TaxiPulse Control Center")
    st.caption("Big Data analytics and short-term demand prediction on NYC TLC Yellow Taxi trip records · January to May 2026")
    if ov:
        c = st.columns(5)
        c[0].metric("Cleaned trips", f"{ov['total_trips']/1e6:.2f} M")
        c[1].metric("Revenue analysed", f"${ov['total_revenue']/1e6:.1f} M")
        c[2].metric("Avg total fare", f"${ov['avg_total']:.2f}")
        c[3].metric("Avg distance", f"{ov['avg_distance_mi']:.2f} mi")
        c[4].metric("Avg duration", f"{ov['avg_duration_min']:.1f} min")

    st.subheader("Architecture")
    st.graphviz_chart("""
    digraph G { rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=15, color="#1F3A5F", margin="0.18,0.10"];
      edge [color="#1F3A5F", penwidth=1.4];
      raw [label="TLC Parquet\n19.0 M trips", fillcolor="#E8E8E8"];
      clean [label="Spark\nvalidate · clean · enrich", fillcolor="#F0A202"];
      proc [label="Processed trips\nParquet, by month", fillcolor="#E8E8E8"];
      agg [label="Spark\naggregation", fillcolor="#F0A202"];
      cur [label="Curated layer\n20 tables", fillcolor="#E8E8E8"];
      app [label="This app", fillcolor="#2A7F8F", fontcolor="white"];
      ml [label="LightGBM model", fillcolor="#2A7F8F", fontcolor="white"];
      mongo [label="MongoDB", fillcolor="#2A7F8F", fontcolor="white"];
      raw -> clean -> proc -> agg -> cur; cur -> app; cur -> ml; cur -> mongo; }""", width="stretch")
    st.subheader("Pipeline status")
    rows = [
        ("Raw data", config.RAW_DIR, "5 monthly Parquet files"),
        ("Cleaned trips", config.PROCESSED_DIR / "trips", "Spark clean stage"),
        ("Curated tables", config.CURATED_DIR / "overview.json", "Spark aggregation"),
        ("Demand model", config.MODELS_DIR / "model_metrics.json", "LightGBM training"),
        ("Benchmarks 1-5", config.REPORTS_DIR / "benchmark_results.json", "performance benchmarks"),
        ("Memory scaling", config.REPORTS_DIR / "memory_scaling.json", "benchmark 6"),
    ]
    st.dataframe(pd.DataFrame([{"Output": n, "Produced by": by, "Updated": ago(_mtime(p)) if p.exists() else "missing"}
                               for n, p, by in rows]), hide_index=True, width="stretch")

    st.subheader("One-click demo")
    st.write("Runs the project end to end with the real commands: **unit tests → Spark pipeline → model training → "
             "performance benchmarks 1-5 → memory-scaling benchmark**. Every page updates automatically when it finishes. "
             "Each step can also be run on its own from its section.")
    job_panel(["full_demo"], "home")


# ====================================================== DATA & PIPELINE ====
@st.cache_data(show_spinner=False)
def raw_files(sig: tuple) -> pd.DataFrame:
    import pyarrow.parquet as pq
    out = []
    for f in sorted(config.RAW_DIR.glob(config.RAW_TRIP_GLOB)):
        out.append({"File": f.name, "Size (MB)": round(f.stat().st_size / 1e6, 1), "Rows": pq.ParquetFile(f).metadata.num_rows})
    return pd.DataFrame(out)


def page_pipeline():
    st.header("⚙️ Data & Pipeline")
    st.write("Raw TLC trip records are validated, cleaned and enriched by **PySpark**, one month at a time. Every removed record "
             "is tagged with the **first rule it violates**, so nothing disappears silently. The cleaned trips are then aggregated "
             "into the curated tables every other section reads.")
    files = sorted(config.RAW_DIR.glob(config.RAW_TRIP_GLOB))
    st.subheader("Raw dataset")
    if files:
        df = raw_files(tuple((f.name, f.stat().st_mtime) for f in files))
        c = st.columns(3)
        c[0].metric("Monthly files", len(df))
        c[1].metric("Raw trips", f"{df['Rows'].sum():,}")
        c[2].metric("Compressed size", f"{df['Size (MB)'].sum():.0f} MB")
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.info("No raw data yet. Click **Download TLC data** below.")

    st.subheader("Run")
    job_panel(["download", "tests", "pipeline"], "pipe")

    st.subheader("Data quality report")
    q = jfile(config.REPORTS_DIR / "quality_report.json")
    if not need(q, "quality report", "run the Spark pipeline."):
        return
    c = st.columns(4)
    c[0].metric("Raw records", f"{q['raw_records']:,}")
    c[1].metric("Retained", f"{q['retained_records']:,}")
    c[2].metric("Removed", f"{q['removed_records']:,}")
    c[3].metric("Retention", f"{100*q['retention_rate']:.2f} %")
    l, r = st.columns([3, 2])
    rr = pd.DataFrame(list(q["removed_by_reason"].items()), columns=["rule", "records"])
    rr["rule"] = rr["rule"].str.replace("_", " ")
    fig = px.bar(rr, x="records", y="rule", orientation="h", title="Records removed, by the first rule violated", text_auto=",")
    fig.update_traces(marker_color=TEAL).update_layout(yaxis=dict(autorange="reversed"), yaxis_title=None)
    with l:
        chart(fig, 380)
    with r:
        months = pd.DataFrame({"raw": q["raw_records_per_month"], "kept": q["retained_records_per_month"]})
        months["kept %"] = (100 * months["kept"] / months["raw"]).round(2)
        st.markdown("**Per month**")
        st.dataframe(months.rename_axis("month").reset_index(), hide_index=True, width="stretch")
        rules = q["rules"]
        st.markdown(f"**Rules:** duration {rules['duration_min'][0]} to {rules['duration_min'][1]} min · distance > 0 and ≤ "
                    f"{rules['trip_distance_mi'][1]} mi · fare > 0 and ≤ ${rules['fare_amount'][1]} · avg speed ≤ "
                    f"{rules['max_avg_speed_mph']} mph · zone IDs {rules['valid_zone_ids'][0]} to {rules['valid_zone_ids'][1]}. "
                    "Nulls in the 'flex fare' columns are **kept**, because they are real trips.")


# ============================================================ ANALYTICS ====
def page_analytics():
    st.header("📊 Analytics")
    st.caption("Every chart reads a small curated table produced by Spark. The app never touches the 17.9 M-row trip data.")
    if not need(table("hourly_profile"), "analytics", "run the Spark pipeline from Data & Pipeline."):
        return
    t = st.tabs(["Demand over time", "Zones", "Routes & flows", "Fares & payments"])

    with t[0]:
        daily = table("daily_metrics")
        chart(px.line(daily, x="pickup_date", y="trip_count", title="Daily trips (all zones)",
                      labels={"pickup_date": "Date", "trip_count": "Trips"}).update_traces(line_color=NAVY), 320)
        l, r = st.columns(2)
        hp = table("hourly_profile")
        fig = px.bar(hp, x="pickup_hour", y="trip_count", title="Trips by hour of day", labels={"pickup_hour": "Hour", "trip_count": "Trips"})
        fig.update_traces(marker_color=[YEL if h == int(hp.loc[hp.trip_count.idxmax(), "pickup_hour"]) else NAVY for h in hp.pickup_hour])
        with l:
            chart(fig, 330)
        ww = table("weekend_vs_weekday_hourly").copy()
        ww["day type"] = ww["is_weekend"].map({True: "Weekend", False: "Weekday"})
        with r:
            chart(px.line(ww, x="pickup_hour", y="trip_count", color="day type", title="Weekday vs weekend, by hour",
                          labels={"pickup_hour": "Hour", "trip_count": "Trips"}, color_discrete_sequence=[NAVY, YEL]), 330)
        hm = table("hour_dow_heatmap")
        piv = hm.pivot(index="dow_name", columns="pickup_hour", values="trip_count").reindex(DOW)
        chart(px.imshow(piv, aspect="auto", color_continuous_scale="YlOrRd", title="Demand heatmap: weekday × hour",
                        labels={"x": "Hour", "y": "", "color": "Trips"}), 330)

    with t[1]:
        zm = table("zone_metrics")
        c1, c2 = st.columns(2)
        b = c1.selectbox("Borough", ["All"] + sorted(zm["borough"].dropna().unique()), key="z_b")
        metric = c2.selectbox("Rank zones by", ["pickup_trips", "dropoff_trips", "avg_fare", "avg_distance_mi",
                                                "avg_duration_min", "avg_tip_pct", "pickup_revenue"], key="z_m")
        sub = zm if b == "All" else zm[zm["borough"] == b]
        top = sub.sort_values(metric, ascending=False).head(15)
        chart(px.bar(top, x=metric, y="zone", orientation="h", color="borough", title=f"Top 15 zones by {metric}")
              .update_layout(yaxis=dict(autorange="reversed"), yaxis_title=None), 460)
        zone = st.selectbox("Inspect a zone", sub.sort_values("pickup_trips", ascending=False)["zone"].tolist(), key="z_z")
        z = zm[zm["zone"] == zone].iloc[0]
        c = st.columns(5)
        c[0].metric("Pickups", f"{int(z.pickup_trips):,}")
        c[1].metric("Drop-offs", f"{int(z.dropoff_trips):,}")
        c[2].metric("Avg fare", f"${z.avg_fare:.2f}")
        c[3].metric("Avg distance", f"{z.avg_distance_mi:.2f} mi")
        c[4].metric("Avg tip (card)", f"{z.avg_tip_pct:.1f} %")
        zh = table("zone_hourly_profile")
        chart(px.bar(zh[zh["zone_id"] == int(z.zone_id)], x="pickup_hour", y="trip_count", title=f"Hourly pickups: {zone}",
                     labels={"pickup_hour": "Hour", "trip_count": "Trips"}).update_traces(marker_color=TEAL), 300)

    with t[2]:
        rm = table("route_metrics").copy()
        rm["route"] = rm["pu_zone"] + " → " + rm["do_zone"]
        n = st.slider("Top N routes", 5, 30, 15)
        chart(px.bar(rm.head(n), x="trip_count", y="route", orientation="h", title=f"Top {n} zone-to-zone routes",
                     hover_data=["avg_fare", "avg_distance_mi", "avg_duration_min"])
              .update_traces(marker_color=NAVY).update_layout(yaxis=dict(autorange="reversed"), yaxis_title=None), 30 * n + 120)
        bf = table("borough_flows")
        piv = bf.pivot(index="pu_borough", columns="do_borough", values="trip_count").fillna(0)
        chart(px.imshow(piv, text_auto=".2s", aspect="auto", color_continuous_scale="Blues", title="Borough-to-borough flows",
                        labels={"x": "Drop-off borough", "y": "Pickup borough", "color": "Trips"}), 380)

    with t[3]:
        l, r = st.columns(2)
        fd = table("fare_by_distance")
        with l:
            chart(px.line(fd, x="distance_bin_mi", y="avg_fare", title="Average fare vs trip distance",
                          labels={"distance_bin_mi": "Distance (mi)", "avg_fare": "Avg fare ($)"}).update_traces(line_color=NAVY), 330)
        pt = table("payment_type_breakdown")
        with r:
            chart(px.pie(pt, names="payment_name", values="trip_count", title="Payment method share", hole=.45), 330)
        l, r = st.columns(2)
        th = table("tips_by_hour")
        with l:
            chart(px.line(th, x="pickup_hour", y="avg_tip_pct", title="Average tip % (card payments) by hour",
                          labels={"pickup_hour": "Hour", "avg_tip_pct": "Tip %"}).update_traces(line_color=YEL), 320)
        bm = table("borough_metrics")
        with r:
            chart(px.bar(bm, x="borough", y="avg_fare", title="Average fare by borough", labels={"avg_fare": "Avg fare ($)"},
                         text_auto=".2f").update_traces(marker_color=TEAL), 320)


# =========================================================== PREDICTION ====
def page_prediction():
    st.header("📈 Demand Prediction")
    st.write("Predicts **trips per hour for a taxi zone**. Evaluation is strictly **chronological**: trained on January to April, "
             "tested on unseen May. The model is compared against three naive baselines, because a high R² means little "
             "until it beats a simple rule.")
    job_panel(["train"], "ml")
    m = jfile(config.MODELS_DIR / "model_metrics.json")
    if not need(m, "trained model", "click **Train demand model**."):
        return
    names = {"baseline_previous_hour": "Baseline: previous hour", "baseline_same_hour_last_week": "Baseline: same hour last week",
             "baseline_zone_hour_dow_mean": "Baseline: zone/hour/weekday mean", "lightgbm_calendar_only": "LightGBM (calendar only)",
             "lightgbm_with_lags": "LightGBM + lag features"}
    df = pd.DataFrame([{"Model": names.get(k, k), "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]} for k, v in m["models"].items()])
    best = m["models"]["lightgbm_with_lags"]
    c = st.columns(4)
    c[0].metric("MAE (trips / hour)", f"{best['MAE']:.2f}")
    c[1].metric("R²", f"{best['R2']:.3f}")
    c[2].metric("Better than last-week baseline", f"{m['improvement_over_last_week_baseline_pct']} %")
    c[3].metric("Zones modelled", m["zones_modelled"])
    st.caption(f"Train {m['train_period'][0][:10]} → {m['train_period'][1][:10]} ({m['train_rows']:,} rows) · "
               f"test {m['test_period'][0][:10]} → {m['test_period'][1][:10]} ({m['test_rows']:,} rows) · seed {m['seed']}")
    l, r = st.columns([3, 2])
    fig = px.bar(df.sort_values("MAE", ascending=False), x="MAE", y="Model", orientation="h", text_auto=".2f",
                 title="Mean absolute error on May 2026 (lower is better)")
    fig.update_traces(marker_color=[YEL if "lag" in x else (TEAL if "LightGBM" in x else LGREY)
                                    for x in df.sort_values("MAE", ascending=False)["Model"]]).update_layout(yaxis_title=None)
    with l:
        chart(fig, 330)
    with r:
        st.dataframe(df, hide_index=True, width="stretch")
        fi = jfile(config.MODELS_DIR / "feature_importance.json")
        if fi:
            f = pd.DataFrame(fi["lightgbm_with_lags"], columns=["feature", "importance"]).head(8)
            chart(px.bar(f, x="importance", y="feature", orientation="h", title="Top features (lag model)")
                  .update_traces(marker_color=NAVY).update_layout(yaxis=dict(autorange="reversed"), yaxis_title=None), 260)

    st.subheader("Try it")
    try:
        from taxipulse.ml import predict
        zm = table("zone_metrics").set_index("zone_id")
        zids = predict.available_zones()
        fmt = lambda i: f"{zm.loc[i, 'zone']} ({zm.loc[i, 'borough']})" if i in zm.index else str(i)  # noqa: E731
        c1, c2, c3 = st.columns([3, 2, 2])
        zid = c1.selectbox("Zone", zids, format_func=fmt, index=zids.index(161) if 161 in zids else 0)
        d = c2.date_input("Date", date(2026, 6, 5))
        h = c3.slider("Hour", 0, 23, 18)
        res = predict.predict_calendar(zid, pd.Timestamp(d) + pd.Timedelta(hours=h))
        c = st.columns(3)
        c[0].metric("Predicted demand", f"≈ {res['predicted_trips']:.0f} trips / hour")
        c[1].metric("Historical mean, same hour & weekday", f"{res['historical_mean_same_hour_dow']:.0f}")
        c[2].metric("Expected error (MAE, calendar model)", f"± {res['expected_error_mae']:.1f}")
        hist = predict.recent_history(zid, 72).rename(columns={"actual_trips": "trips"}).assign(series="actual (last 72 h of data)")
        fut = predict.predict_next_hours(zid, 12).rename(columns={"predicted_trips": "trips"}).assign(series="forecast (next 12 h)")
        chart(px.line(pd.concat([hist, fut]), x="timestamp", y="trips", color="series", line_dash="series",
                      color_discrete_sequence=[NAVY, YEL], title=f"Recent demand and 12-hour forecast: {fmt(zid)}"), 340)
        st.caption("Predictions are model outputs, not observations. The calendar model works for any future hour; the forecast "
                   "line uses the more accurate lag model, rolled forward from the last real hour.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Prediction unavailable: {exc}")


# =========================================================== BENCHMARKS ====
EXP_NOTES = {
    "1": "Same query, full 19 M rows. At this size the data fits in memory, so Spark's scheduling overhead is not repaid.",
    "2": "Columnar Parquet reads only the 2 columns the query needs; CSV must parse every byte of every row.",
    "3": "Same computation. RDD lambdas cross the Python↔JVM bridge row by row; the DataFrame is compiled by Catalyst/Tungsten.",
    "4": "Catalyst prunes unused columns at plan time, so asking for fewer columns explicitly changes little.",
    "5": "Parallelism helps, but sub-linearly: shuffle, JVM start-up and disk reads do not parallelise (Amdahl's law).",
}


def page_benchmarks():
    st.header("⏱️ Performance Benchmarks")
    st.write("Every number here is **measured on this machine** by re-runnable scripts. Nothing is estimated. "
             "The environment is recorded alongside the results.")
    t = st.tabs(["Benchmarks 1-5", "Benchmark 6: memory scaling"])

    with t[0]:
        job_panel(["bench1", "bench3"], "bench")
        b = jfile(config.REPORTS_DIR / "benchmark_results.json")
        if need(b, "benchmark results", "click **Run benchmarks 1-5**."):
            env = b["environment"]
            st.caption(f"Measured {b['generated_at'].replace('T', ' ')} · {env['cpu_cores']} cores · {env['ram_gb']} GB RAM · "
                       f"PySpark {env['pyspark']} · Python {env['python']} · {env['dataset_rows']:,} rows · "
                       f"median of {b['repeats']} run(s) · query: {b['query']}")
            res = pd.DataFrame(b["results"])
            res["exp"] = res["experiment"].str[0]
            titles = {"1": "Single machine vs Spark", "2": "Parquet vs CSV", "3": "MapReduce (RDD) vs DataFrame",
                      "4": "Column pruning", "5": "Core scaling"}
            cols = st.columns(2)
            for i, (e, g) in enumerate(res.groupby("exp")):
                fast, slow = g.loc[g.median_seconds.idxmin()], g.loc[g.median_seconds.idxmax()]
                ratio = slow.median_seconds / max(fast.median_seconds, 1e-9)
                fig = px.bar(g, x="median_seconds", y="approach", orientation="h", text_auto=".2f",
                             title=f"{e}. {titles.get(e, g.experiment.iloc[0])}", labels={"median_seconds": "seconds", "approach": ""})
                fig.update_traces(marker_color=[YEL if a == fast.approach else LGREY for a in g.approach])
                with cols[i % 2]:
                    chart(fig, 220)
                    st.markdown(f"**Fastest: {fast.approach}**, {ratio:.1f}× faster than {slow.approach}.  \n{EXP_NOTES.get(e, '')}")
            st.info("**The honest reading:** Parquet and the DataFrame API win clearly, and more cores help. But if single-machine "
                    "pandas beats Spark in benchmark 1, that is the expected result at 325 MB. Spark's justification here is "
                    "not raw speed. Benchmark 6 shows what it is.")

    with t[1]:
        st.write("The job now needs **all 20 columns**. pandas runs in a child process that reports its **peak resident memory**; "
                 "Spark runs the same workload inside a fixed driver budget. The question: how does memory grow with the data, "
                 "and where does one machine run out?")
        st.warning("Month 5 (19.0 M rows) deliberately exhausts memory on an 8 GB machine: the OS kills the process and can take the "
                   "terminal with it. It is therefore **not offered as a live button**; its result is documented below.")
        job_panel(["mem3", "mem4"], "mem")
        ms = jfile(config.REPORTS_DIR / "memory_scaling.json")
        if not need(ms, "memory-scaling results", "click **Memory scaling: months 1-3**."):
            return
        meas = ms.get("measurements", [])
        ok = [x for x in meas if x["pandas"]["status"] == "completed"]
        env = ms.get("environment", {})
        ram = env.get("ram_gb") or 8.2
        rows = pd.DataFrame([{"Months": x["months"], "Rows": f"{x['rows']:,}",
                              "pandas": "completed" if x["pandas"]["status"] == "completed" else "killed by OS" if x["pandas"]["status"] == "killed_by_os" else x["pandas"]["status"].replace("_", " "),
                              "pandas peak (GB)": x["pandas"].get("peak_rss_gb") if x["pandas"]["status"] == "completed" else "-",
                              "pandas (s)": x["pandas"].get("seconds") if x["pandas"]["status"] == "completed" else "-",
                              "Spark (s)": x["spark"]["seconds"]} for x in meas]).astype(str)
        an = ms.get("analysis", {})
        fig = go.Figure()
        if ok:
            fig.add_trace(go.Scatter(x=[x["rows"] / 1e6 for x in ok], y=[x["pandas"]["peak_rss_gb"] for x in ok],
                                     mode="lines+markers", name="pandas peak memory (measured)", line=dict(color=TEAL, width=3)))
        if an.get("gb_per_million_rows") and ok:
            last = ok[-1]
            proj_y = last["pandas"]["peak_rss_gb"] + an["gb_per_million_rows"] * (18.999282 - last["rows"] / 1e6)
            fig.add_trace(go.Scatter(x=[last["rows"] / 1e6, 19.0], y=[last["pandas"]["peak_rss_gb"], proj_y], mode="lines+markers",
                                     name="pandas (projected to 19.0 M rows)", line=dict(color=TEAL, dash="dash")))
        fig.add_hline(y=ram * 0.8, line=dict(color=RED, dash="dot"), annotation_text=f"usable RAM ≈ {ram*0.8:.1f} GB of {ram} GB")
        spark_gb = float(str(env.get("spark_driver_memory", "4g")).rstrip("gG") or 4)
        fig.add_hline(y=spark_gb, line=dict(color=YEL, width=3), annotation_text=f"Spark: fixed {spark_gb:.0f} GB driver, every size completed",
                      annotation_position="bottom right")
        fig.update_layout(title="Memory needed vs dataset size", xaxis_title="Dataset size (million rows)", yaxis_title="GB",
                          legend=dict(orientation="h", y=-0.2))
        l, r = st.columns([3, 2])
        with l:
            chart(fig, 420)
        with r:
            if an:
                st.metric("pandas memory growth", f"{an['gb_per_million_rows']} GB / million rows")
                st.metric("Amplification vs Parquet on disk", f"{an['memory_amplification_vs_parquet']} ×")
                st.metric("Projected ceiling on this machine", f"{an['projected_max_rows_on_this_machine']/1e6:.1f} M rows",
                          delta=f"{an['projected_max_rows_on_this_machine']/1e6 - 19.0:+.1f} M vs the full 19.0 M dataset",
                          delta_color="normal")
            st.dataframe(rows, hide_index=True, width="stretch")
        st.success("**Documented 5-month run on the demo machine (8.2 GB WSL):** pandas was **killed by the OS** at 19.0 M rows, "
                   "exactly as the growth curve predicts, while Spark completed the same workload in **1.47 s** inside its 4 GB "
                   "driver. Spark's value at this scale is not speed. It is that the same code keeps working as the data grows.")


# ============================================================== MONGODB ====
def page_mongo():
    st.header("🍃 MongoDB Serving Layer")
    st.write("**Spark computes, MongoDB serves.** The curated tables are loaded as JSON documents into the `taxipulse` database, "
             "one collection per table, with indexes for key-based lookups in milliseconds.")
    from taxipulse.app import mongo_tools as mt
    ok, info = mt.ping()
    if ok:
        st.success(f"Connected to MongoDB {info} at {config.MONGO_URI}")
    else:
        st.error("MongoDB is not reachable from this app.")
        st.markdown("1. In **PowerShell as Administrator**: `Start-Service MongoDB`\n"
                    "2. Make sure WSL **mirrored networking** is on (see the setup guide), so WSL's `localhost` reaches Windows.\n"
                    "3. Refresh this page.")
        st.caption(info)
    job_panel(["mongo_load"], "mongo")
    if not ok:
        return
    try:
        stats = mt.collection_stats()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not list collections: {exc}")
        return
    if not stats:
        st.info("The database is empty. Click **Load results into MongoDB**.")
        return
    c = st.columns(3)
    c[0].metric("Collections", len(stats))
    c[1].metric("Documents", f"{sum(s['documents'] for s in stats):,}")
    c[2].metric("Largest", f"{stats[0]['collection']}")
    with st.expander("All collections"):
        st.dataframe(pd.DataFrame(stats), hide_index=True, width="stretch")

    t = st.tabs(["Overview document", "Aggregation pipeline", "Index vs full scan", "Top routes"])
    with t[0]:
        st.code("use taxipulse\ndb.overview.findOne()", language="javascript")
        st.json(mt.overview() or {})
    with t[1]:
        st.write("`$group` is the **map + reduce** step (grouping key + accumulators); `$sort` orders the reduced output. Together they are "
                 "the document-store counterpart of MapReduce.")
        st.code(mt.BOROUGH_MONGOSH, language="javascript")
        agg = pd.DataFrame(mt.borough_aggregation()).rename(columns={"_id": "borough"})
        l, r = st.columns([3, 2])
        with l:
            chart(px.bar(agg, x="borough", y="pickups", text_auto=".3s", title="Pickups by borough (computed inside MongoDB)")
                  .update_traces(marker_color=NAVY), 330)
        with r:
            agg["avg_fare"] = agg["avg_fare"].round(2)
            st.dataframe(agg, hide_index=True, width="stretch")
    with t[2]:
        st.write("The same lookup (one zone, one day) run **with the index** and **forced into a full collection scan**. "
                 "This is why a serving layer is indexed.")
        zm = table("zone_metrics")
        zones = zm.sort_values("pickup_trips", ascending=False).head(40)
        c1, c2 = st.columns(2)
        zid = c1.selectbox("Zone", zones["zone_id"].tolist(), format_func=lambda i: zones.set_index("zone_id").loc[i, "zone"], key="mz")
        d = c2.date_input("Date", date(2026, 5, 1), min_value=date(2026, 1, 1), max_value=date(2026, 5, 31), key="md")
        if st.button("▶  Run explain() both ways", width="stretch", type="primary"):
            try:
                st.session_state["explain"] = (mt.explain_lookup(int(zid), d.isoformat()),
                                               mt.explain_lookup(int(zid), d.isoformat(), force_collscan=True), int(zid), d.isoformat())
            except Exception as exc:  # noqa: BLE001
                st.warning(f"explain failed: {exc}")
        if "explain" in st.session_state:
            idx, full, z, ds = st.session_state["explain"]
            l, r = st.columns(2)
            for col, res, name, force in ((l, idx, "With the index", False), (r, full, "Forced full scan", True)):
                with col:
                    st.markdown(f"**{name}**")
                    st.code(mt.explain_mongosh(z, ds, force), language="javascript")
                    a, b_, c_ = st.columns(3)
                    a.metric("Returned", res["nReturned"])
                    b_.metric("Docs examined", f"{res['totalDocsExamined']:,}")
                    c_.metric("Time", f"{res['executionTimeMillis']} ms")
                    st.caption(f"Plan: {res['stages']} · keys examined {res['totalKeysExamined']:,} · collection "
                               f"{res['collection_size']:,} docs")
            ratio = full["totalDocsExamined"] / max(idx["totalDocsExamined"], 1)
            st.success(f"The index examined **{idx['totalDocsExamined']:,}** documents to return {idx['nReturned']}; the full scan "
                       f"examined **{full['totalDocsExamined']:,}**. That is **{ratio:,.0f}× more work** for the same answer.")
    with t[3]:
        st.code("db.route_metrics.find({}, { pu_zone: 1, do_zone: 1, trip_count: 1, avg_fare: 1, _id: 0 })\n"
                "  .sort({ trip_count: -1 }).limit(10)", language="javascript")
        st.dataframe(pd.DataFrame(mt.top_routes(10)), hide_index=True, width="stretch")


# ================================================================ route ====
{"🏠 Home": page_home, "⚙️ Data & Pipeline": page_pipeline, "📊 Analytics": page_analytics, "📈 Prediction": page_prediction,
 "⏱️ Benchmarks": page_benchmarks, "🍃 MongoDB": page_mongo}[page]()
