import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pandas as pd

from monitoring.fault_detector import CSV_PATH, classify_sample, compute_thresholds
from optimization.fault_injection import add_latency, remove_fault
from optimization.self_healing import (
    build_network_graph,
    clear_route,
    get_current_path,
    reroute_traffic,
    self_heal,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "optimization_results.csv"
RESULT_COLUMNS = [
    "timestamp",
    "fault_type",
    "source",
    "destination",
    "original_path",
    "fault_link",
    "alternate_path",
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
    "rerouting_success",
]
SOURCE = "h1"
DESTINATION = "h4"
FAULT_LINK = ("s1", "s2")
FAULT_TYPE = "latency_50ms"
LATENCY_MS = 50
IPERF_DURATION_SECONDS = 3
PING_COUNT = 5


def _require_root_and_mininet():
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        raise PermissionError(
            "Mininet and OVS require root. Run: "
            "sudo python3 -m optimization.evaluate_optimization"
        )
    try:
        from mininet.net import Mininet
        from mininet.node import OVSBridge
        from network.topology import DigitalTwinTopo
    except ImportError as error:
        raise RuntimeError(
            "Mininet Python modules are not installed for this Python interpreter. "
            "Install the system Mininet/OVS packages and run with sudo."
        ) from error
    return Mininet, OVSBridge, DigitalTwinTopo


def _disable_stp_for_explicit_routes(network):
    for switch in network.switches:
        result = switch.cmd(
            "ovs-vsctl",
            "set",
            "bridge",
            switch.name,
            "stp_enable=false",
            "rstp_enable=false",
        )
        if "error" in result.lower():
            raise RuntimeError(f"Could not disable STP on {switch.name}: {result}")
    time.sleep(2)


def _start_iperf_server(server_host):
    server = server_host.popen(
        ["iperf3", "-s"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            if server.poll() is not None:
                raise RuntimeError("iperf3 server exited before becoming ready")
            listeners = server_host.cmd("ss", "-H", "-ltn", "sport", "=", ":5201")
            if listeners.strip():
                return server
            time.sleep(0.1)
        raise RuntimeError("iperf3 server did not listen on port 5201")
    except BaseException:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=3)
        raise


def _communicate_process(process, timeout, label):
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process.kill()
        stdout, stderr = process.communicate()
        raise RuntimeError(
            f"{label} timed out. " + (stderr or stdout or "")
        ) from error
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.communicate()
        raise
    return stdout, stderr


def _measure_ping(source_host, destination_ip):
    process = source_host.popen(
        [
            "ping",
            "-n",
            "-c",
            str(PING_COUNT),
            "-W",
            "2",
            destination_ip,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output, _ = _communicate_process(process, PING_COUNT * 2 + 5, "ping")
    if process.returncode not in (0, 1):
        raise RuntimeError(f"ping failed with status {process.returncode}:\n{output}")
    loss_match = re.search(r"(\d+(?:\.\d+)?)%\s+packet loss", output)
    rtt_match = re.search(
        r"(?:rtt|round-trip)\s+min/avg/max/(?:mdev|stddev)\s*=\s*"
        r"([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s*ms",
        output,
    )
    if not loss_match:
        raise RuntimeError(f"Could not parse ping packet loss:\n{output}")
    packet_loss = float(loss_match.group(1))
    if rtt_match:
        average_rtt = float(rtt_match.group(2))
        jitter = float(rtt_match.group(4))
    else:
        average_rtt = None
        jitter = None
    return {
        "packet_loss": packet_loss,
        "avg_rtt": average_rtt,
        "jitter": jitter,
        "ping_output": output,
    }


def _measure_throughput(source_host, destination_ip):
    client = source_host.popen(
        [
            "iperf3",
            "-c",
            destination_ip,
            "-t",
            str(IPERF_DURATION_SECONDS),
            "-J",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = _communicate_process(
        client,
        IPERF_DURATION_SECONDS + 10,
        "iperf3 client",
    )
    if client.returncode != 0:
        raise RuntimeError(f"iperf3 failed: {stderr or stdout}")
    try:
        iperf_result = json.loads(stdout)
        bits_per_second = iperf_result["end"]["sum_received"]["bits_per_second"]
        retransmissions = iperf_result["end"]["sum_sent"].get("retransmits")
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"Could not parse iperf3 JSON: {stdout}") from error
    if isinstance(bits_per_second, bool) or not isinstance(
        bits_per_second, (int, float)
    ):
        raise RuntimeError("iperf3 receiver bits_per_second was not numeric")
    return {
        "throughput_mbps": bits_per_second / 1_000_000,
        "retransmissions": retransmissions,
    }


def measure_network(network, source=SOURCE, destination=DESTINATION):
    source_host = network.get(source)
    destination_host = network.get(destination)
    ping_metrics = _measure_ping(source_host, destination_host.IP())
    throughput_metrics = _measure_throughput(source_host, destination_host.IP())
    return {**ping_metrics, **throughput_metrics}


def _load_detector_thresholds():
    telemetry = pd.read_csv(CSV_PATH)
    baseline = telemetry[
        telemetry["fault_type"].fillna("").astype(str).str.strip().str.lower()
        == "baseline"
    ].copy()
    if baseline.empty:
        raise ValueError(f"No baseline rows found in {CSV_PATH}")
    for feature in (
        "packet_loss",
        "avg_rtt",
        "jitter",
        "throughput_mbps",
        "retransmissions",
    ):
        baseline[feature] = pd.to_numeric(baseline[feature], errors="coerce")
    return compute_thresholds(baseline)


def _detect_fault(metrics, thresholds):
    return classify_sample(
        pd.Series(
            {
                "packet_loss": metrics["packet_loss"],
                "avg_rtt": metrics["avg_rtt"],
                "jitter": metrics["jitter"],
                "throughput_mbps": metrics["throughput_mbps"],
                "retransmissions": metrics["retransmissions"],
            }
        ),
        thresholds,
    )


def _append_result(row):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
    if not write_header:
        with RESULTS_PATH.open("r", newline="", encoding="utf-8") as result_file:
            existing_columns = next(csv.reader(result_file), [])
        if existing_columns != RESULT_COLUMNS:
            raise ValueError(
                f"Unexpected columns in existing {RESULTS_PATH}; it was not changed"
            )
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _format_path(path):
    return " -> ".join(path) if path else "Unavailable"


def _print_metrics(title, path, metrics):
    print(f"\n---------- {title} ----------")
    print(f"Path: {_format_path(path)}")
    print(f"Throughput: {metrics['throughput_mbps']:.3f} Mbps")
    print(f"RTT: {metrics['avg_rtt'] if metrics['avg_rtt'] is not None else 'N/A'} ms")
    print(f"Packet Loss: {metrics['packet_loss']:.2f}%")


def _percentage_change(before, after, lower_is_better=False):
    if before is None or after is None or before == 0:
        return None
    if lower_is_better:
        return (before - after) / abs(before) * 100
    return (after - before) / abs(before) * 100


def _notify_stage(
    callback,
    stage,
    message,
    current_path=None,
    metrics=None,
    status="running",
    original_path=None,
    recovered_path=None,
    decision_time=None,
    recovery_time=None,
    rerouting_success=None,
    baseline_metrics=None,
    fault_metrics=None,
    after_metrics=None,
    detected_fault=None,
):
    if callback is not None:
        callback(
            stage=stage,
            status=status,
            message=message,
            current_path=current_path,
            fault_type=FAULT_TYPE if stage not in {"STARTING", "NORMAL"} else None,
            fault_link=(
                f"{FAULT_LINK[0]} -> {FAULT_LINK[1]}"
                if stage not in {"STARTING", "NORMAL"}
                else None
            ),
            metrics=metrics or {},
            original_path=original_path,
            recovered_path=recovered_path,
            decision_time=decision_time,
            recovery_time=recovery_time,
            rerouting_success=rerouting_success,
            baseline_metrics=baseline_metrics,
            fault_metrics=fault_metrics,
            after_metrics=after_metrics,
            detected_fault=detected_fault,
        )


def run_experiment(stage_callback=None):
    Mininet, OVSBridge, DigitalTwinTopo = _require_root_and_mininet()
    _notify_stage(
        stage_callback,
        "STARTING",
        "Starting the Mininet topology and preparing the controlled experiment.",
    )
    thresholds = _load_detector_thresholds()
    network = Mininet(
        topo=DigitalTwinTopo(),
        switch=OVSBridge,
        controller=None,
        autoSetMacs=True,
        autoStaticArp=True,
        build=True,
    )
    server = None
    fault_applied = False
    network_started = False

    try:
        network.start()
        network_started = True
        _disable_stp_for_explicit_routes(network)

        baseline_path = ["h1", "s1", "s2", "s4", "h4"]
        if not nx_path_is_valid(build_network_graph(network), baseline_path):
            raise RuntimeError(f"Configured baseline path is not live: {baseline_path}")
        reroute_traffic(network, SOURCE, DESTINATION, baseline_path)
        print(f"SOURCE: {SOURCE} ({network.get(SOURCE).IP()})")
        print(f"DESTINATION: {DESTINATION} ({network.get(DESTINATION).IP()})")
        print(f"FAULT LINK: {FAULT_LINK[0]}-{FAULT_LINK[1]}")
        server = _start_iperf_server(network.get(DESTINATION))

        baseline = measure_network(network)
        _print_metrics("BASELINE", baseline_path, baseline)
        _notify_stage(
            stage_callback,
            "NORMAL",
            "Baseline traffic measured on the normal route.",
            current_path=baseline_path,
            metrics=baseline,
            original_path=baseline_path,
            baseline_metrics=baseline,
        )

        fault_started = time.perf_counter()
        add_latency(network, *FAULT_LINK, LATENCY_MS)
        fault_applied = True
        _notify_stage(
            stage_callback,
            "FAULT_INJECTED",
            f"Injected {LATENCY_MS} ms latency on {FAULT_LINK[0]} -> {FAULT_LINK[1]}.",
            current_path=baseline_path,
            metrics=baseline,
            original_path=baseline_path,
            baseline_metrics=baseline,
        )
        fault_metrics = measure_network(network)
        detection_started = time.perf_counter()
        detected_status = _detect_fault(fault_metrics, thresholds)
        detection_time = time.perf_counter() - detection_started
        _print_metrics("FAULT", baseline_path, fault_metrics)
        print(f"Fault Type: {FAULT_TYPE} ({LATENCY_MS} ms injected latency)")
        print(f"Affected Link: {FAULT_LINK[0]}-{FAULT_LINK[1]}")
        print(f"Telemetry detector classification: {detected_status}")
        _notify_stage(
            stage_callback,
            "FAULT_DETECTED",
            f"Statistical telemetry detector classified the sample as {detected_status}.",
            current_path=baseline_path,
            metrics=fault_metrics,
            original_path=baseline_path,
            baseline_metrics=baseline,
            fault_metrics=fault_metrics,
            detected_fault=detected_status,
        )

        _notify_stage(
            stage_callback,
            "SELF_HEALING",
            "Selecting an alternate path and updating Open vSwitch forwarding.",
            current_path=baseline_path,
            metrics=fault_metrics,
            original_path=baseline_path,
            baseline_metrics=baseline,
            fault_metrics=fault_metrics,
            detected_fault=detected_status,
        )
        if detected_status == "NORMAL":
            healing = {
                "original_path": baseline_path,
                "alternate_path": None,
                "decision_time": detection_time,
                "success": False,
                "reason": "Telemetry detector did not classify the sample as a fault",
            }
        else:
            healing = self_heal(
                network,
                SOURCE,
                DESTINATION,
                failed_link=FAULT_LINK,
            )
        _notify_stage(
            stage_callback,
            "REROUTED",
            (
                "Open vSwitch forwarding now uses the alternate path."
                if healing["success"]
                else f"Rerouting failed: {healing['reason']}"
            ),
            current_path=(
                healing["alternate_path"]
                if healing["success"]
                else healing["original_path"]
            ),
            metrics=fault_metrics,
            status="running" if healing["success"] else "failed",
            original_path=healing["original_path"],
            recovered_path=healing["alternate_path"],
            decision_time=healing["decision_time"],
            rerouting_success=healing["success"],
            baseline_metrics=baseline,
            fault_metrics=fault_metrics,
            detected_fault=detected_status,
        )
        recovery_time = time.perf_counter() - fault_started
        print("\n---------- SELF-HEALING ----------")
        print(f"Original Path: {_format_path(healing['original_path'])}")
        print(f"Alternate Path: {_format_path(healing['alternate_path'])}")
        print(f"Decision Time: {healing['decision_time'] * 1000:.3f} ms")
        print(f"Recovery Time: {recovery_time:.3f} s")
        print(f"Rerouting Status: {'SUCCESS' if healing['success'] else 'FAILED'}")
        if healing["reason"]:
            print(f"Reason: {healing['reason']}")

        after_path = (
            healing["alternate_path"] if healing["success"] else healing["original_path"]
        )
        after_metrics = measure_network(network)
        _print_metrics("AFTER OPTIMIZATION", after_path, after_metrics)
        _notify_stage(
            stage_callback,
            "RECOVERED",
            (
                "Post-reroute network performance measured."
                if healing["success"]
                else "Recovery measurement completed, but rerouting was unsuccessful."
            ),
            current_path=after_path,
            metrics=after_metrics,
            status="running" if healing["success"] else "failed",
            original_path=healing["original_path"],
            recovered_path=after_path if healing["success"] else None,
            decision_time=healing["decision_time"],
            recovery_time=recovery_time,
            rerouting_success=healing["success"],
            baseline_metrics=baseline,
            fault_metrics=fault_metrics,
            after_metrics=after_metrics,
            detected_fault=detected_status,
        )

        throughput_change = _percentage_change(
            fault_metrics["throughput_mbps"],
            after_metrics["throughput_mbps"],
        )
        rtt_change = _percentage_change(
            fault_metrics["avg_rtt"],
            after_metrics["avg_rtt"],
            lower_is_better=True,
        )
        loss_change = None
        if fault_metrics["packet_loss"] is not None and after_metrics["packet_loss"] is not None:
            loss_change = fault_metrics["packet_loss"] - after_metrics["packet_loss"]

        print("\n---------- PERFORMANCE CHANGE ----------")
        print(
            "Throughput improvement: "
            + (f"{throughput_change:+.2f}%" if throughput_change is not None else "N/A")
        )
        print(
            "RTT improvement: "
            + (f"{rtt_change:+.2f}%" if rtt_change is not None else "N/A")
        )
        print(
            "Packet-loss reduction: "
            + (f"{loss_change:+.2f} percentage points" if loss_change is not None else "N/A")
        )

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fault_type": FAULT_TYPE,
            "source": SOURCE,
            "destination": DESTINATION,
            "original_path": _format_path(healing["original_path"]),
            "fault_link": f"{FAULT_LINK[0]}-{FAULT_LINK[1]}",
            "alternate_path": _format_path(healing["alternate_path"]),
            "throughput_before": baseline["throughput_mbps"],
            "throughput_fault": fault_metrics["throughput_mbps"],
            "throughput_after": after_metrics["throughput_mbps"],
            "rtt_before": baseline["avg_rtt"],
            "rtt_fault": fault_metrics["avg_rtt"],
            "rtt_after": after_metrics["avg_rtt"],
            "packet_loss_before": baseline["packet_loss"],
            "packet_loss_fault": fault_metrics["packet_loss"],
            "packet_loss_after": after_metrics["packet_loss"],
            "decision_time": healing["decision_time"],
            "recovery_time": recovery_time,
            "rerouting_success": healing["success"],
        }
        _append_result(row)
        print(f"\nExperiment appended to {RESULTS_PATH}")
        _notify_stage(
            stage_callback,
            "COMPLETED",
            (
                "Experiment completed and results appended."
                if healing["success"]
                else "Experiment completed; rerouting was unsuccessful."
            ),
            current_path=after_path,
            metrics=after_metrics,
            status="completed" if healing["success"] else "failed",
            original_path=healing["original_path"],
            recovered_path=after_path if healing["success"] else None,
            decision_time=healing["decision_time"],
            recovery_time=recovery_time,
            rerouting_success=healing["success"],
            baseline_metrics=baseline,
            fault_metrics=fault_metrics,
            after_metrics=after_metrics,
            detected_fault=detected_status,
        )
        return row
    finally:
        cleanup_errors = []
        if fault_applied:
            try:
                remove_fault(network, *FAULT_LINK)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                cleanup_errors.append(f"tc cleanup: {error}")
        if server is not None:
            try:
                server.terminate()
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=3)
            except OSError as error:
                cleanup_errors.append(f"iperf3 cleanup: {error}")
        if network_started:
            try:
                clear_route(network, SOURCE, DESTINATION)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                cleanup_errors.append(f"OVS flow cleanup: {error}")
        try:
            network.stop()
        except (OSError, RuntimeError) as error:
            cleanup_errors.append(f"Mininet cleanup: {error}")
        if cleanup_errors:
            print("Cleanup issues: " + "; ".join(cleanup_errors), file=sys.stderr)


def nx_path_is_valid(graph, path):
    return all(graph.has_edge(left, right) for left, right in zip(path, path[1:]))


def main():
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\nExperiment interrupted; cleaning up Mininet and injected faults.")
    except (PermissionError, RuntimeError, ValueError, OSError) as error:
        print(f"Optimization experiment failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
