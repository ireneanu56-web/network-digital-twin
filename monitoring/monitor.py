import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import re
import tempfile
import time


CSV_COLUMNS = (
    "timestamp",
    "fault_type",
    "packet_loss",
    "avg_rtt",
    "jitter",
    "throughput_mbps",
    "retransmissions",
)
OLD_CSV_COLUMNS = CSV_COLUMNS[:5]


def mininet_pid(host):
    result = subprocess.run(
        ["pgrep", "-f", f"mininet:{host}"],
        capture_output=True,
        text=True,
        check=False,
    )
    process_ids = result.stdout.splitlines()
    if result.returncode != 0 or not process_ids:
        raise RuntimeError(f"Could not find the Mininet {host} process")
    return process_ids[0]


def measure_ping(host="10.0.0.4", count=10):
    """
    Run ping from Mininet host h1 to h4.
    """

    command = [
        "sudo",
        "mnexec",
        "-a",
        mininet_pid("h1"),
        "ping",
        "-c",
        str(count),
        host
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    output = result.stdout

    print(output)

    # Extract packet loss
    loss_match = re.search(
        r"(\d+(?:\.\d+)?)% packet loss",
        output
    )

    # Extract RTT statistics
    rtt_match = re.search(
        r"rtt min/avg/max/mdev = "
        r"([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)",
        output
    )

    packet_loss = (
        float(loss_match.group(1))
        if loss_match
        else None
    )

    if rtt_match:
        avg_rtt = float(rtt_match.group(2))
        jitter = float(rtt_match.group(4))
    else:
        avg_rtt = None
        jitter = None

    return {
        "packet_loss": packet_loss,
        "avg_rtt": avg_rtt,
        "jitter": jitter
    }


def stop_iperf_server(host_pid):
    result = subprocess.run(
        [
            "sudo",
            "mnexec",
            "-a",
            host_pid,
            "pkill",
            "-x",
            "iperf3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Could not stop iperf3 processes on h4: {result.stderr}")


def iperf_server_listening(host_pid):
    result = subprocess.run(
        [
            "sudo",
            "mnexec",
            "-a",
            host_pid,
            "ss",
            "-H",
            "-ltn",
            "sport",
            "=",
            ":5201",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not check the h4 iperf3 port: {result.stderr}")
    return bool(result.stdout.strip())


def start_iperf_server(host_pid):
    start_result = subprocess.run(
        ["sudo", "mnexec", "-a", host_pid, "iperf3", "-s", "-D"],
        capture_output=True,
        text=True,
        check=False,
    )
    if start_result.returncode != 0:
        raise RuntimeError(
            f"Could not start iperf3 server on h4: {start_result.stderr}"
        )

    for _ in range(30):
        if iperf_server_listening(host_pid):
            print("Fresh iperf3 server on h4 is ready.")
            return
        time.sleep(0.1)
    raise RuntimeError("iperf3 server on h4 did not listen on port 5201")


def ensure_iperf_server():
    host_pid = mininet_pid("h4")
    stop_iperf_server(host_pid)
    try:
        start_iperf_server(host_pid)
    except Exception:
        stop_iperf_server(host_pid)
        raise
    return host_pid


def measure_throughput(host="10.0.0.4", duration=5):
    ensure_iperf_server()
    try:
        result = subprocess.run(
            [
                "sudo",
                "mnexec",
                "-a",
                mininet_pid("h1"),
                "iperf3",
                "-c",
                host,
                "-t",
                str(duration),
                "-J",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired as error:
        print("iperf3 client timed out after 15 seconds.")
        if error.stdout:
            print(error.stdout)
        if error.stderr:
            print(error.stderr)
        raise RuntimeError("iperf3 client timed out after 15 seconds") from error

    raw_output = result.stdout + result.stderr
    if result.returncode != 0:
        print("iperf3 client failed; raw iperf3 output:")
        print(raw_output)
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )

    try:
        iperf_data = json.loads(result.stdout)
        receiver_bits_per_second = iperf_data["end"]["sum_received"]["bits_per_second"]
        retransmissions = iperf_data["end"]["sum_sent"]["retransmits"]
        if isinstance(receiver_bits_per_second, bool) or not isinstance(
            receiver_bits_per_second, (int, float)
        ):
            raise TypeError("bits_per_second must be numeric")
        if isinstance(retransmissions, bool) or not isinstance(
            retransmissions, (int, float)
        ):
            raise TypeError("retransmits must be numeric")
        throughput_mbps = receiver_bits_per_second / 1_000_000
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print("Could not parse iperf3 JSON; raw iperf3 output:")
        print(raw_output)
        raise RuntimeError(
            "iperf3 output did not contain valid JSON metrics"
        ) from error

    print(f"JSON receiver bits/sec: {receiver_bits_per_second}")
    print(f"Parsed throughput: {throughput_mbps} Mbps")
    print(f"Parsed retransmissions: {retransmissions}")

    return {
        "throughput_mbps": throughput_mbps,
        "retransmissions": retransmissions,
    }


def prepare_csv(csv_path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return True

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        existing_columns = reader.fieldnames
        if existing_columns == list(CSV_COLUMNS):
            return False
        if existing_columns != list(OLD_CSV_COLUMNS):
            raise ValueError(f"Unexpected columns in {csv_path}")
        old_rows = list(reader)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            newline="",
            encoding="utf-8",
            dir=csv_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(old_rows)
        os.replace(temporary_path, csv_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fault_type", nargs="?", default="baseline")
    parser.add_argument("samples", nargs="?", type=int, default=1)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be a positive integer")

    print("Network Digital Twin - Monitoring")
    print("----------------------------------")

    data_directory = Path(__file__).resolve().parent.parent / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    csv_path = data_directory / "network_data.csv"

    write_header = prepare_csv(csv_path)

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for sample_number in range(1, args.samples + 1):
            print(f"\nSample {sample_number}/{args.samples}")
            print("Measuring network from Mininet h1...")

            metrics = measure_ping()
            throughput_runs = [measure_throughput() for _ in range(3)]
            throughput_mbps = statistics.median(
                run["throughput_mbps"] for run in throughput_runs
            )
            retransmissions = statistics.median(
                run["retransmissions"] for run in throughput_runs
            )
            print("Throughput measurements:")
            for run_number, run in enumerate(throughput_runs, start=1):
                print(f"  Run {run_number}: {run['throughput_mbps']} Mbps")
            print(f"  Median: {throughput_mbps} Mbps")
            print("Retransmission measurements:")
            for run_number, run in enumerate(throughput_runs, start=1):
                print(f"  Run {run_number}: {run['retransmissions']}")
            print(f"  Median: {retransmissions}")

            timestamp = datetime.now(timezone.utc).isoformat()
            writer.writerow({
                "timestamp": timestamp,
                "fault_type": args.fault_type,
                "packet_loss": metrics["packet_loss"],
                "avg_rtt": metrics["avg_rtt"],
                "jitter": metrics["jitter"],
                "throughput_mbps": throughput_mbps,
                "retransmissions": retransmissions,
            })
            csv_file.flush()

            print("Measurement saved to data/network_data.csv")
            print(f"Timestamp   : {timestamp}")
            print(f"Fault Type  : {args.fault_type}")
            print(f"Packet Loss : {metrics['packet_loss']}%")
            print(f"Average RTT : {metrics['avg_rtt']} ms")
            print(f"Jitter      : {metrics['jitter']} ms")
            print(f"Throughput  : {throughput_mbps} Mbps")
            print(f"Retransmits : {retransmissions}")

            if sample_number < args.samples:
                time.sleep(2)