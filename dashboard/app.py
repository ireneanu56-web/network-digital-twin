from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.fault_detector import (  # noqa: E402
    CSV_PATH,
    FAULT_MAP,
    classify_sample,
    compute_thresholds,
)


METRICS = {
    "packet_loss": ("Packet Loss", "%"),
    "avg_rtt": ("Average RTT", "ms"),
    "jitter": ("Jitter", "ms"),
    "throughput_mbps": ("Throughput", "Mbps"),
    "retransmissions": ("Retransmissions", ""),
}
FAULT_CLASSES = ["NORMAL", "LATENCY_FAULT", "PACKET_LOSS_FAULT"]


def format_metric(value, unit=""):
    if pd.isna(value):
        return "N/A"
    if unit == "%":
        return f"{value:.2f}{unit}"
    if unit:
        return f"{value:,.2f} {unit}"
    return f"{value:,.0f}"


def format_optimization_value(value, unit="", decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}{unit}"


@st.cache_data(ttl=5)
def load_data(csv_path):
    return pd.read_csv(csv_path)


@st.fragment(run_every=1)
def render_live_demo():
    status_path = PROJECT_ROOT / "results" / "live_demo_status.json"
    st.header("🔴 LIVE NETWORK DIGITAL TWIN DEMO")

    if not status_path.exists():
        st.info("Live demo is not currently running.")
        return

    try:
        with status_path.open("r", encoding="utf-8") as status_file:
            live_status = json.load(status_file)
        if not isinstance(live_status, dict):
            st.warning("Live demo status is not a JSON object.")
            return
    except (OSError, json.JSONDecodeError) as error:
        st.warning(f"Could not read live demo status: {error}")
        return

    stage = str(live_status.get("stage") or "UNKNOWN").replace("_", " ")
    status = str(live_status.get("status") or "unknown").lower()
    st.subheader(f"Current Stage: {stage}")
    message = live_status.get("message") or "Waiting for experiment status."
    if stage == "ERROR" or status in {"error", "interrupted"}:
        st.error(message)
    elif status == "failed":
        st.warning(message)
    elif status == "completed":
        if live_status.get("rerouting_success") is True:
            st.success("Network successfully recovered through alternate routing.")
        else:
            st.warning(message)
    else:
        st.info(message)

    metrics = live_status.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    current_path = live_status.get("current_path") or "Not available yet"
    st.markdown(f"### Current Path: {current_path}")
    detail_columns = st.columns(2)
    detail_columns[0].markdown(
        f"**Fault Type:** {live_status.get('fault_type') or 'N/A'}"
    )
    detail_columns[1].markdown(
        f"**Fault Link:** {live_status.get('fault_link') or 'N/A'}"
    )
    st.markdown(
        f"**Original Path:** {live_status.get('original_path') or 'Not available yet'}"
    )
    st.markdown(
        f"**Recovered Path:** {live_status.get('recovered_path') or 'Not available yet'}"
    )

    live_metrics = [
        ("Throughput", "throughput_mbps", "Mbps"),
        ("RTT", "avg_rtt", "ms"),
        ("Packet Loss", "packet_loss", "%"),
        ("Jitter", "jitter", "ms"),
        ("Retransmissions", "retransmissions", ""),
    ]
    metric_columns = st.columns(len(live_metrics))
    for column, (label, key, unit) in zip(metric_columns, live_metrics):
        value = pd.to_numeric(metrics.get(key), errors="coerce")
        if pd.isna(value):
            display_value = "N/A"
        elif unit == "%":
            display_value = f"{value:.2f}{unit}"
        elif unit:
            display_value = f"{value:,.2f} {unit}"
        else:
            display_value = f"{value:,.0f}"
        column.metric(label, display_value)

    metric_states = live_status.get("metric_states")
    if not isinstance(metric_states, dict):
        metric_states = {}

    def format_live_cell(value, unit):
        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric_value):
            return "N/A"
        if unit == "%":
            return f"{numeric_value:.2f}"
        return f"{numeric_value:,.2f}"

    comparison = pd.DataFrame(
        [
            {
                "Metric": "Throughput (Mbps)",
                "Normal": format_live_cell(
                    metric_states.get("throughput_before"), "Mbps"
                ),
                "During Fault": format_live_cell(
                    metric_states.get("throughput_fault"), "Mbps"
                ),
                "After Self-Healing": format_live_cell(
                    metric_states.get("throughput_after"), "Mbps"
                ),
            },
            {
                "Metric": "RTT (ms)",
                "Normal": format_live_cell(metric_states.get("rtt_before"), "ms"),
                "During Fault": format_live_cell(
                    metric_states.get("rtt_fault"), "ms"
                ),
                "After Self-Healing": format_live_cell(
                    metric_states.get("rtt_after"), "ms"
                ),
            },
            {
                "Metric": "Packet Loss (%)",
                "Normal": format_live_cell(
                    metric_states.get("packet_loss_before"), "%"
                ),
                "During Fault": format_live_cell(
                    metric_states.get("packet_loss_fault"), "%"
                ),
                "After Self-Healing": format_live_cell(
                    metric_states.get("packet_loss_after"), "%"
                ),
            },
        ]
    )
    st.markdown("**Live Before / Fault / After**")
    st.dataframe(comparison, hide_index=True, width="stretch")

    rerouting_success = live_status.get("rerouting_success")
    status_columns = st.columns(2)
    decision_time = pd.to_numeric(live_status.get("decision_time"), errors="coerce")
    recovery_time = pd.to_numeric(live_status.get("recovery_time"), errors="coerce")
    status_columns[0].metric(
        "Decision Time",
        "N/A" if pd.isna(decision_time) else f"{decision_time * 1000:.2f} ms",
    )
    status_columns[1].metric(
        "Recovery Time",
        "N/A" if pd.isna(recovery_time) else f"{recovery_time:.2f} s",
    )
    if rerouting_success is True:
        st.success("Rerouting Status: SUCCESS")
    elif rerouting_success is False:
        st.error("Rerouting Status: FAILURE")

    st.markdown(
        "**Live experiment sequence:**  \n"
        "NORMAL → FAULT DETECTED → SELF-HEALING → REROUTED → RECOVERED"
    )

    timestamp = live_status.get("timestamp")
    if timestamp:
        st.caption(f"Status updated: {timestamp}")
    st.caption(
        "Live experiment: Mininet generates network traffic, telemetry is collected, "
        "a controlled fault is injected, the fault detector identifies the "
        "degradation, and the self-healing module reroutes traffic through an "
        "alternate path."
    )


def main():
    st.set_page_config(
        page_title="Network Digital Twin Dashboard",
        page_icon="🌐",
        layout="wide",
    )
    st.title("Network Digital Twin Dashboard")
    st.caption("Live view of the latest recorded telemetry and experiment history.")
    render_live_demo()

    try:
        data = load_data(str(CSV_PATH))
    except FileNotFoundError:
        st.error(f"Telemetry CSV was not found: {CSV_PATH}")
        st.stop()
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load telemetry CSV: {error}")
        st.stop()

    required_columns = {"timestamp", "fault_type", *METRICS}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        st.error(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")
        st.stop()
    if data.empty:
        st.warning("The telemetry CSV has no samples yet.")
        st.stop()

    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")

    data["timestamp_parsed"] = pd.to_datetime(
        data["timestamp"], errors="coerce", utc=True
    )
    baseline_mask = data["fault_type"].fillna("").astype(str).str.strip().str.lower() == "baseline"
    baseline_data = data.loc[baseline_mask]
    if baseline_data.empty:
        st.error("No baseline samples were found; fault thresholds cannot be calculated.")
        st.stop()

    thresholds = compute_thresholds(baseline_data)
    data["predicted_status"] = data.apply(
        lambda row: classify_sample(row, thresholds), axis=1
    )
    if data["timestamp_parsed"].notna().any():
        latest_index = data["timestamp_parsed"].idxmax()
    else:
        latest_index = data.index[-1]
    latest = data.loc[latest_index]
    latest_status = latest["predicted_status"]
    latest_time = latest["timestamp"]

    st.subheader("Latest Network Status")
    status_text = f"**{latest_status}**"
    if latest_status == "NORMAL":
        st.success(status_text)
    elif latest_status == "LATENCY_FAULT":
        st.warning(status_text)
    else:
        st.error(status_text)
    st.caption(f"Based on the latest recorded sample: {latest_time}")

    metric_columns = st.columns(5)
    for column, (key, (label, unit)) in zip(metric_columns, METRICS.items()):
        column.metric(label, format_metric(latest[key], unit))

    st.subheader("Fault Detection Summary")
    status_counts = data["predicted_status"].value_counts().reindex(
        FAULT_CLASSES, fill_value=0
    )
    fault_count_columns = st.columns(3)
    fault_count_labels = [
        ("NORMAL", "Normal Samples"),
        ("LATENCY_FAULT", "Latency Faults"),
        ("PACKET_LOSS_FAULT", "Packet Loss Faults"),
    ]
    for column, (status, label) in zip(fault_count_columns, fault_count_labels):
        column.metric(label, int(status_counts[status]))

    actual_status = data["fault_type"].apply(
        lambda fault_type: FAULT_MAP.get(
            str(fault_type).strip().lower(), "NORMAL"
        )
    )
    detection_accuracy = (actual_status == data["predicted_status"]).mean()
    st.markdown(f"**Detection Accuracy: {detection_accuracy:.2%}**")

    chart_data = data.copy()
    if chart_data["timestamp_parsed"].notna().any():
        chart_data = chart_data.sort_values("timestamp_parsed", na_position="last")
    chart_data = chart_data.reset_index(drop=True)
    chart_data.insert(0, "Sample", range(1, len(chart_data) + 1))

    st.subheader("Telemetry Trends")
    chart_columns = st.columns(2)
    chart_specs = [
        ("avg_rtt", "RTT over samples", chart_columns[0]),
        ("throughput_mbps", "Throughput over samples", chart_columns[1]),
        ("packet_loss", "Packet loss over samples", chart_columns[0]),
        ("retransmissions", "Retransmissions over samples", chart_columns[1]),
    ]
    for metric, title, column in chart_specs:
        with column:
            st.markdown(f"**{title}**")
            st.line_chart(chart_data, x="Sample", y=metric)

    st.subheader("Experiment Condition Comparison")
    comparison_labels = {
        "NORMAL": "baseline",
        "LATENCY_FAULT": "latency_50ms",
        "PACKET_LOSS_FAULT": "packet_loss_2.5pct",
    }
    comparison_rows = []
    actual_conditions = (
        data["fault_type"].fillna("").astype(str).str.strip().str.lower()
    )
    for label, source_fault in comparison_labels.items():
        condition_rows = data.loc[actual_conditions == source_fault]
        row = {
            "Condition": label,
            "Samples": len(condition_rows),
        }
        for metric, (display_name, unit) in METRICS.items():
            mean_value = condition_rows[metric].mean()
            row[display_name] = format_metric(mean_value, unit)
        comparison_rows.append(row)
    st.dataframe(
        pd.DataFrame(comparison_rows),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Recent Telemetry Samples")
    recent = chart_data.tail(10).copy()
    recent["Detected Status"] = recent["predicted_status"]
    recent = recent.rename(
        columns={
            "timestamp": "Timestamp",
            "fault_type": "Experiment",
            "packet_loss": "Packet Loss (%)",
            "avg_rtt": "Average RTT (ms)",
            "jitter": "Jitter (ms)",
            "throughput_mbps": "Throughput (Mbps)",
            "retransmissions": "Retransmissions",
        }
    )
    recent_columns = [
        "Timestamp",
        "Experiment",
        "Detected Status",
        "Packet Loss (%)",
        "Average RTT (ms)",
        "Jitter (ms)",
        "Throughput (Mbps)",
        "Retransmissions",
    ]
    st.dataframe(
        recent[recent_columns].iloc[::-1],
        hide_index=True,
        width="stretch",
    )

    st.subheader("ML Fault Detection Comparison")
    ml_results_path = PROJECT_ROOT / "results" / "ml_results.csv"
    if not ml_results_path.exists():
        st.info(
            "ML results are not available yet. Train and evaluate the models with "
            "`python -m models.train_models` and `python -m models.evaluate_models`."
        )
    else:
        try:
            ml_results = pd.read_csv(ml_results_path)
            required_result_columns = {
                "Evaluation ID",
                "Model",
                "Task",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "Detection Time (ms/sample)",
            }
            missing_result_columns = required_result_columns.difference(
                ml_results.columns
            )
            if missing_result_columns:
                st.warning(
                    "ML results have an unsupported format: "
                    + ", ".join(sorted(missing_result_columns))
                )
            elif ml_results.empty:
                st.info("The ML results file contains no evaluated model rows yet.")
            else:
                latest_evaluation = ml_results["Evaluation ID"].max()
                latest_results = ml_results[
                    ml_results["Evaluation ID"] == latest_evaluation
                ].copy()
                display_columns = [
                    "Model",
                    "Task",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-score",
                    "Detection Time (ms/sample)",
                ]
                st.dataframe(
                    latest_results[display_columns],
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Autoencoder metrics measure binary normal/anomaly detection; "
                    "the other models report three-class fault classification. "
                    "The latest run uses chronological holdout samples."
                )
                st.markdown("**Latest evaluation accuracy comparison**")
                st.bar_chart(
                    latest_results.set_index("Model")[["Accuracy"]],
                    y="Accuracy",
                    width="stretch",
                )
        except (OSError, pd.errors.ParserError) as error:
            st.warning(f"Could not load saved ML comparison results: {error}")

    st.subheader("Performance Optimization / Self-Healing")
    optimization_results_path = PROJECT_ROOT / "results" / "optimization_results.csv"
    st.markdown(
        "After detecting a degraded link, the Network Digital Twin identifies an "
        "alternate path and updates forwarding so traffic can avoid the affected "
        "link. Network performance is measured before the fault, during the fault, "
        "and after rerouting to quantify recovery."
    )
    st.info(
        "Current optimization implementation is a rule-based self-healing prototype "
        "using shortest-path/alternate-path selection and Open vSwitch forwarding "
        "changes. It is not reinforcement learning."
    )
    optimization_plot_names = [
        "self_healing_throughput.png",
        "self_healing_rtt.png",
        "self_healing_packet_loss.png",
        "self_healing_recovery_time.png",
    ]
    available_optimization_plots = [
        PROJECT_ROOT / "results" / "plots" / name
        for name in optimization_plot_names
        if (PROJECT_ROOT / "results" / "plots" / name).exists()
    ]
    if (
        not optimization_results_path.exists()
        or optimization_results_path.stat().st_size == 0
    ):
        st.info("No self-healing experiment results available yet.")
    else:
        try:
            optimization_results = pd.read_csv(optimization_results_path)
            required_optimization_columns = {
                "timestamp",
                "fault_type",
                "fault_link",
                "original_path",
                "alternate_path",
                "rerouting_success",
                "decision_time",
                "recovery_time",
                "throughput_before",
                "throughput_fault",
                "throughput_after",
                "rtt_before",
                "rtt_fault",
                "rtt_after",
                "packet_loss_before",
                "packet_loss_fault",
                "packet_loss_after",
            }
            missing_optimization_columns = required_optimization_columns.difference(
                optimization_results.columns
            )
            if missing_optimization_columns:
                st.warning(
                    "Self-healing results have an unsupported format: "
                    + ", ".join(sorted(missing_optimization_columns))
                )
                optimization_results = pd.DataFrame()
            if optimization_results.empty:
                st.info("No self-healing experiment results available yet.")
            else:
                numeric_optimization_columns = [
                    "decision_time",
                    "recovery_time",
                    "throughput_before",
                    "throughput_fault",
                    "throughput_after",
                    "rtt_before",
                    "rtt_fault",
                    "rtt_after",
                    "packet_loss_before",
                    "packet_loss_fault",
                    "packet_loss_after",
                ]
                for column in numeric_optimization_columns:
                    optimization_results[column] = pd.to_numeric(
                        optimization_results[column],
                        errors="coerce",
                    )
                optimization_results["timestamp_parsed"] = pd.to_datetime(
                    optimization_results["timestamp"],
                    errors="coerce",
                    utc=True,
                )
                if optimization_results["timestamp_parsed"].notna().any():
                    latest_optimization = optimization_results.loc[
                        optimization_results["timestamp_parsed"].idxmax()
                    ]
                else:
                    latest_optimization = optimization_results.iloc[-1]

                is_success = (
                    str(latest_optimization["rerouting_success"]).strip().lower()
                    in {"true", "1", "yes", "success"}
                )
                st.markdown(
                    f"**Fault condition:** {latest_optimization['fault_type']}"
                )
                st.markdown(f"**Fault link:** {latest_optimization['fault_link']}")
                st.markdown(
                    f"**Original path:** {latest_optimization['original_path']}"
                )
                st.markdown(
                    f"**Alternate path:** {latest_optimization['alternate_path']}"
                )
                decision_time = latest_optimization.get("decision_time")
                recovery_time = latest_optimization.get("recovery_time")
                status_columns = st.columns(3)
                status_columns[0].metric(
                    "Decision Time",
                    format_optimization_value(
                        decision_time * 1000 if pd.notna(decision_time) else None,
                        " ms",
                    ),
                )
                status_columns[1].metric(
                    "Recovery Time",
                    format_optimization_value(recovery_time, " s"),
                )
                if is_success:
                    status_columns[2].success("SUCCESS — successful rerouting")
                else:
                    status_columns[2].error("FAILURE — unsuccessful rerouting")

                st.subheader("Self-Healing Before/Fault/After Metrics")
                latest_metric_rows = [
                    {
                        "Metric": "Throughput (Mbps)",
                        "Baseline": latest_optimization["throughput_before"],
                        "During Fault": latest_optimization["throughput_fault"],
                        "After Self-Healing": latest_optimization["throughput_after"],
                    },
                    {
                        "Metric": "RTT (ms)",
                        "Baseline": latest_optimization["rtt_before"],
                        "During Fault": latest_optimization["rtt_fault"],
                        "After Self-Healing": latest_optimization["rtt_after"],
                    },
                    {
                        "Metric": "Packet Loss (%)",
                        "Baseline": latest_optimization["packet_loss_before"],
                        "During Fault": latest_optimization["packet_loss_fault"],
                        "After Self-Healing": latest_optimization["packet_loss_after"],
                    },
                ]
                latest_metric_table = pd.DataFrame(latest_metric_rows)
                for column in (
                    "Baseline",
                    "During Fault",
                    "After Self-Healing",
                ):
                    latest_metric_table[column] = latest_metric_table[column].map(
                        lambda value: format_optimization_value(value)
                    )
                st.dataframe(
                    latest_metric_table,
                    hide_index=True,
                    width="stretch",
                )

                success_values = (
                    optimization_results["rerouting_success"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"true", "1", "yes", "success"})
                )
                success_count = int(success_values.sum())
                experiment_count = len(optimization_results)
                success_rate = success_count / experiment_count * 100

                st.subheader("Self-Healing Summary Across Experiments")
                summary_rows = [
                    ("Total Experiments", f"{experiment_count:,}"),
                    ("Successful Reroutes", f"{success_count:,}"),
                    ("Rerouting Success Rate", f"{success_rate:.2f}%"),
                    (
                        "Mean Decision Time",
                        format_optimization_value(
                            optimization_results["decision_time"].mean() * 1000,
                            " ms",
                        ),
                    ),
                    (
                        "Mean Recovery Time",
                        format_optimization_value(
                            optimization_results["recovery_time"].mean(),
                            " s",
                        ),
                    ),
                    (
                        "Mean Throughput Before",
                        format_optimization_value(
                            optimization_results["throughput_before"].mean(),
                            " Mbps",
                        ),
                    ),
                    (
                        "Mean Throughput During Fault",
                        format_optimization_value(
                            optimization_results["throughput_fault"].mean(),
                            " Mbps",
                        ),
                    ),
                    (
                        "Mean Throughput After Healing",
                        format_optimization_value(
                            optimization_results["throughput_after"].mean(),
                            " Mbps",
                        ),
                    ),
                    (
                        "Mean RTT Before",
                        format_optimization_value(
                            optimization_results["rtt_before"].mean(),
                            " ms",
                        ),
                    ),
                    (
                        "Mean RTT During Fault",
                        format_optimization_value(
                            optimization_results["rtt_fault"].mean(),
                            " ms",
                        ),
                    ),
                    (
                        "Mean RTT After Healing",
                        format_optimization_value(
                            optimization_results["rtt_after"].mean(),
                            " ms",
                        ),
                    ),
                ]
                for start in range(0, len(summary_rows), 4):
                    summary_columns = st.columns(4)
                    for column, (label, value) in zip(
                        summary_columns,
                        summary_rows[start : start + 4],
                    ):
                        column.metric(label, value)

                if available_optimization_plots:
                    st.subheader("Optimization Plots")
                    for plot_path in available_optimization_plots:
                        st.image(
                            str(plot_path),
                            caption=plot_path.stem.replace("_", " ").title(),
                        )
                else:
                    st.caption(
                        "Optimization plot files are not available yet. Generate them with "
                        "`python3 -m optimization.plot_results`."
                    )
        except (OSError, pd.errors.ParserError, KeyError) as error:
            st.warning(f"Could not load self-healing results: {error}")

    with st.expander("Baseline thresholds used by fault detection"):
        st.caption("These are computed by monitoring/fault_detector.py from baseline rows.")
        st.json(thresholds)

    st.subheader("System Pipeline")
    pipeline_columns = st.columns(5)
    pipeline_steps = [
        "Mininet",
        "Monitoring",
        "Telemetry CSV",
        "Fault Detection",
        "Dashboard",
    ]
    for column, step in zip(pipeline_columns, pipeline_steps):
        column.markdown(
            f"<div style='text-align:center; padding:0.8rem; "
            f"border:1px solid #d9e2f2; border-radius:0.6rem; "
            f"background:#f5f8ff; color:#1f2937 !important; "
            f"-webkit-text-fill-color:#1f2937; opacity:1; font-weight:700'>"
            f"{step}</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
