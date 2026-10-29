import json
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from optimization import evaluate_optimization


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = PROJECT_ROOT / "results" / "live_demo_status.json"
SOURCE = evaluate_optimization.SOURCE
DESTINATION = evaluate_optimization.DESTINATION


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def write_status(
    stage,
    status,
    message,
    current_path=None,
    fault_type=None,
    fault_link=None,
    metrics=None,
    metric_states=None,
    detected_fault=None,
    original_path=None,
    recovered_path=None,
    decision_time=None,
    recovery_time=None,
    rerouting_success=None,
):
    """Atomically publish a JSON-compatible live experiment snapshot."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "stage": stage,
        "status": status,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "destination": DESTINATION,
        "current_path": (
            " → ".join(current_path)
            if isinstance(current_path, (list, tuple))
            else current_path
        ),
        "fault_type": fault_type,
        "detected_fault": detected_fault,
        "fault_link": fault_link,
        "metrics": _json_safe(metrics or {}),
        "metric_states": _json_safe(
            metric_states
            or {
                "throughput_before": None,
                "rtt_before": None,
                "packet_loss_before": None,
                "throughput_fault": None,
                "rtt_fault": None,
                "packet_loss_fault": None,
                "throughput_after": None,
                "rtt_after": None,
                "packet_loss_after": None,
            }
        ),
        "original_path": (
            " → ".join(original_path)
            if isinstance(original_path, (list, tuple))
            else original_path
        ),
        "recovered_path": (
            " → ".join(recovered_path)
            if isinstance(recovered_path, (list, tuple))
            else recovered_path
        ),
        "decision_time": decision_time,
        "recovery_time": recovery_time,
        "rerouting_success": rerouting_success,
    }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=STATUS_PATH.parent,
            prefix=".live_demo_status.",
            suffix=".tmp",
            delete=False,
        ) as status_file:
            temporary_path = Path(status_file.name)
            json.dump(document, status_file, ensure_ascii=False, indent=2)
            status_file.write("\n")
        os.replace(temporary_path, STATUS_PATH)
        sudo_uid = os.environ.get("SUDO_UID")
        sudo_gid = os.environ.get("SUDO_GID")
        if os.geteuid() == 0 and sudo_uid is not None and sudo_gid is not None:
            os.chown(STATUS_PATH, int(sudo_uid), int(sudo_gid))
        os.chmod(STATUS_PATH, 0o644)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return document


def _measurement_values(metrics, suffix):
    if not metrics:
        return {}
    return {
        f"throughput_{suffix}": metrics.get("throughput_mbps"),
        f"rtt_{suffix}": metrics.get("avg_rtt"),
        f"packet_loss_{suffix}": metrics.get("packet_loss"),
    }


def _stage_callback_factory():
    metric_states = {
        "throughput_before": None,
        "rtt_before": None,
        "packet_loss_before": None,
        "throughput_fault": None,
        "rtt_fault": None,
        "packet_loss_fault": None,
        "throughput_after": None,
        "rtt_after": None,
        "packet_loss_after": None,
    }
    latest_fields = {}

    def stage_callback(**event):
        metric_states.update(
            _measurement_values(event.get("baseline_metrics"), "before")
        )
        metric_states.update(
            _measurement_values(event.get("fault_metrics"), "fault")
        )
        metric_states.update(
            _measurement_values(event.get("after_metrics"), "after")
        )
        latest_fields.update(
            {
                key: event[key]
                for key in (
                    "original_path",
                    "recovered_path",
                    "decision_time",
                    "recovery_time",
                    "rerouting_success",
                )
                if event.get(key) is not None
            }
        )
        if event.get("rerouting_success") is not None:
            latest_fields["rerouting_success"] = event["rerouting_success"]

        document = write_status(
            stage=event["stage"],
            status=event["status"],
            message=event["message"],
            current_path=event.get("current_path"),
            fault_type=event.get("fault_type"),
            fault_link=event.get("fault_link"),
            metrics=event.get("metrics"),
            metric_states=metric_states,
            detected_fault=event.get("detected_fault"),
            **latest_fields,
        )
        print(
            f"[LIVE DEMO] {document['stage']}: {document['message']}",
            flush=True,
        )

    return stage_callback


def _previous_status_fields():
    try:
        with STATUS_PATH.open("r", encoding="utf-8") as status_file:
            previous = json.load(status_file)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(previous, dict):
        return {}
    return {
        key: previous[key]
        for key in (
            "current_path",
            "fault_type",
            "fault_link",
            "metrics",
            "metric_states",
            "detected_fault",
            "original_path",
            "recovered_path",
            "decision_time",
            "recovery_time",
            "rerouting_success",
        )
        if key in previous
    }


def main():
    write_status(
        stage="STARTING",
        status="running",
        message="Preparing the Mininet live network demonstration.",
        fault_type=evaluate_optimization.FAULT_TYPE,
        fault_link=(
            f"{evaluate_optimization.FAULT_LINK[0]} -> "
            f"{evaluate_optimization.FAULT_LINK[1]}"
        ),
    )
    print("[LIVE DEMO] STARTING: Preparing the Mininet live network demonstration.", flush=True)
    try:
        evaluate_optimization.run_experiment(
            stage_callback=_stage_callback_factory()
        )
    except KeyboardInterrupt:
        write_status(
            stage="ERROR",
            status="interrupted",
            message="Live experiment was interrupted by the user.",
            **_previous_status_fields(),
        )
        print("[LIVE DEMO] ERROR: Live experiment was interrupted by the user.", flush=True)
        raise SystemExit(130)
    except Exception as error:
        write_status(
            stage="ERROR",
            status="error",
            message=f"{type(error).__name__}: {error}",
            **_previous_status_fields(),
        )
        print(
            f"[LIVE DEMO] ERROR: {type(error).__name__}: {error}",
            flush=True,
        )
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
