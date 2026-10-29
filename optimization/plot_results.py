from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "optimization_results.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def main():
    if not RESULTS_PATH.exists():
        print("No self-healing experiment results available yet.")
        return
    data = pd.read_csv(RESULTS_PATH)
    if data.empty:
        print("No self-healing experiment results available yet.")
        return

    required_columns = {
        "timestamp",
        "fault_type",
        "throughput_before",
        "throughput_fault",
        "throughput_after",
        "rtt_before",
        "rtt_fault",
        "rtt_after",
        "packet_loss_before",
        "packet_loss_fault",
        "packet_loss_after",
        "decision_time",
        "recovery_time",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Optimization results are missing columns: {', '.join(sorted(missing_columns))}"
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.sort_values("timestamp", na_position="last").reset_index(drop=True)
    labels = [
        f"{index + 1}: {row.fault_type}"
        for index, row in data.iterrows()
    ]

    for title, columns, ylabel, filename in [
        (
            "Throughput: baseline, fault, and after self-healing",
            ["throughput_before", "throughput_fault", "throughput_after"],
            "Throughput (Mbps)",
            "self_healing_throughput.png",
        ),
        (
            "RTT: baseline, fault, and after self-healing",
            ["rtt_before", "rtt_fault", "rtt_after"],
            "Average RTT (ms)",
            "self_healing_rtt.png",
        ),
        (
            "Packet loss: baseline, fault, and after self-healing",
            ["packet_loss_before", "packet_loss_fault", "packet_loss_after"],
            "Packet loss (%)",
            "self_healing_packet_loss.png",
        ),
    ]:
        figure, axis = plt.subplots(figsize=(9, 5))
        for column, label in zip(columns, ["Before", "During fault", "After"]):
            axis.plot(labels, data[column], marker="o", label=label)
        axis.set_title(title)
        axis.set_xlabel("Experiment")
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    recovery_axis = axis.twinx()
    positions = range(len(data))
    axis.bar(
        [position - 0.2 for position in positions],
        data["decision_time"] * 1000,
        width=0.4,
        label="Decision time (ms)",
        color="#4c78a8",
    )
    recovery_axis.bar(
        [position + 0.2 for position in positions],
        data["recovery_time"],
        width=0.4,
        label="Recovery time (s)",
        color="#f58518",
    )
    axis.set_xticks(list(positions), labels, rotation=25, ha="right")
    axis.set_title("Self-healing decision and recovery time")
    axis.set_ylabel("Decision time (ms)")
    recovery_axis.set_ylabel("Recovery time (s)")
    handles, legend_labels = axis.get_legend_handles_labels()
    recovery_handles, recovery_labels = recovery_axis.get_legend_handles_labels()
    axis.legend(handles + recovery_handles, legend_labels + recovery_labels)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "self_healing_recovery_time.png", dpi=150)
    plt.close(figure)

    print(f"Generated four plots from {len(data)} recorded experiment(s) in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
