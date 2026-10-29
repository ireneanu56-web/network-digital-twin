from pathlib import Path

import pandas as pd


CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "network_data.csv"
FAULT_MAP = {
    "baseline": "NORMAL",
    "latency_50ms": "LATENCY_FAULT",
    "packet_loss_2.5pct": "PACKET_LOSS_FAULT",
}


def compute_thresholds(baseline_df: pd.DataFrame) -> dict:
    """Set simple, explainable thresholds from baseline behavior."""
    rtt_mean = baseline_df["avg_rtt"].mean()
    rtt_std = baseline_df["avg_rtt"].std(ddof=0)
    rtt_q99 = baseline_df["avg_rtt"].quantile(0.99)
    rtt_threshold = max(rtt_mean + 5 * rtt_std, rtt_q99)

    jitter_mean = baseline_df["jitter"].mean()
    jitter_std = baseline_df["jitter"].std(ddof=0)
    jitter_q99 = baseline_df["jitter"].quantile(0.99)
    jitter_threshold = max(jitter_mean + 5 * jitter_std, jitter_q99)

    retrans_mean = baseline_df["retransmissions"].mean()
    retrans_std = baseline_df["retransmissions"].std(ddof=0)
    retrans_q99 = baseline_df["retransmissions"].quantile(0.99)
    retrans_threshold = max(retrans_mean + 5 * retrans_std, retrans_q99)

    packet_loss_q99 = baseline_df["packet_loss"].quantile(0.99)
    packet_loss_threshold = max(packet_loss_q99, 0.0)

    throughput_q05 = baseline_df["throughput_mbps"].quantile(0.05)

    return {
        "packet_loss_threshold": packet_loss_threshold,
        "avg_rtt_threshold": rtt_threshold,
        "jitter_threshold": jitter_threshold,
        "retransmissions_threshold": retrans_threshold,
        "throughput_low_threshold": throughput_q05,
    }


def classify_sample(row: pd.Series, thresholds: dict) -> str:
    """Rule-based fault classification with one class per sample."""
    packet_loss = pd.to_numeric(row["packet_loss"], errors="coerce") or 0.0
    retransmissions = pd.to_numeric(row["retransmissions"], errors="coerce") or 0.0
    avg_rtt = pd.to_numeric(row["avg_rtt"], errors="coerce")
    jitter = pd.to_numeric(row["jitter"], errors="coerce")
    throughput_mbps = pd.to_numeric(row["throughput_mbps"], errors="coerce")

    packet_loss_rule = packet_loss > thresholds["packet_loss_threshold"]
    retrans_rule = retransmissions > thresholds["retransmissions_threshold"]
    throughput_drop_rule = (
        pd.notna(throughput_mbps)
        and throughput_mbps < thresholds["throughput_low_threshold"]
        and retransmissions > 0.5 * thresholds["retransmissions_threshold"]
    )

    if packet_loss_rule or retrans_rule or throughput_drop_rule:
        return "PACKET_LOSS_FAULT"

    latency_rule = (
        pd.notna(avg_rtt)
        and avg_rtt > thresholds["avg_rtt_threshold"]
    ) or (
        pd.notna(jitter)
        and jitter > thresholds["jitter_threshold"]
        and pd.notna(avg_rtt)
        and avg_rtt > 0.5 * thresholds["avg_rtt_threshold"]
    )

    if latency_rule:
        return "LATENCY_FAULT"

    return "NORMAL"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    baseline_df = df[df["fault_type"].str.lower() == "baseline"].copy()

    if baseline_df.empty:
        raise ValueError(f"No baseline rows found in {CSV_PATH}")

    thresholds = compute_thresholds(baseline_df)

    print("Baseline thresholds used by the detector:")
    for key, value in thresholds.items():
        print(f"  {key}: {value}")

    predictions = []
    actuals = []

    for _, row in df.iterrows():
        actual = FAULT_MAP.get(str(row["fault_type"]).strip().lower(), "NORMAL")
        predicted = classify_sample(row, thresholds)
        actuals.append(actual)
        predictions.append(predicted)

        print(
            f"Actual: {actual:<16} | Predicted: {predicted:<16} | "
            f"packet_loss={row['packet_loss']} | avg_rtt={row['avg_rtt']} | "
            f"jitter={row['jitter']} | throughput_mbps={row['throughput_mbps']} | "
            f"retransmissions={row['retransmissions']}"
        )

    results = pd.DataFrame({
        "actual": actuals,
        "predicted": predictions,
    })

    accuracy = (results["actual"] == results["predicted"]).mean()
    print(f"\nOverall classification accuracy: {accuracy:.2%}")

    labels = ["NORMAL", "LATENCY_FAULT", "PACKET_LOSS_FAULT"]
    confusion = pd.crosstab(results["actual"], results["predicted"]).reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion)


if __name__ == "__main__":
    main()
