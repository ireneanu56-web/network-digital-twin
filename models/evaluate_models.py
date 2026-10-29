import json
from datetime import datetime, timezone
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from models.autoencoder_detector import reconstruction_errors
from models.gnn_detector import load_gnn, predict_gnn
from models.gru_detector import predict_gru
from models.preprocess import prepare_data
from monitoring.fault_detector import classify_sample, compute_thresholds


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_DIR = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULT_COLUMNS = [
    "Evaluation ID",
    "Model",
    "Task",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "Detection Time (ms/sample)",
    "Test Samples",
]


def classification_metrics(actual, predicted, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }


def timed_ms_per_sample(call, sample_count):
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    return result, elapsed * 1000 / max(sample_count, 1)


def evaluate_statistical(data):
    baseline_train = data["raw_train"][
        data["raw_train"]["actual_label"] == "NORMAL"
    ]
    if baseline_train.empty:
        raise ValueError("Training split has no baseline samples for threshold fitting")
    thresholds = compute_thresholds(baseline_train)
    started = time.perf_counter()
    labels = [
        classify_sample(row, thresholds)
        for _, row in data["raw_test"].iterrows()
    ]
    elapsed = time.perf_counter() - started
    predictions = data["label_encoder"].transform(labels)
    return predictions, elapsed * 1000 / len(labels)


def add_plot_suffix(name, evaluation_id):
    return PLOTS_DIR / f"{name}_{evaluation_id}.png"


def save_confusion_plot(actual, predicted, label_names, title, filename):
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=np.arange(len(label_names)),
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(filename, dpi=150)
    plt.close(figure)


def save_comparison_plots(results, evaluation_id, data, ae_errors, ae_threshold):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(results["Model"], results["Accuracy"], color="#4776b4")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Held-out Test Accuracy (Autoencoder is binary)")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(add_plot_suffix("model_accuracy", evaluation_id), dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(results))
    bar_width = 0.24
    for offset, metric, color in [
        (-bar_width, "Precision", "#4c78a8"),
        (0, "Recall", "#f58518"),
        (bar_width, "F1-score", "#54a24b"),
    ]:
        axis.bar(
            positions + offset,
            results[metric],
            bar_width,
            label=metric,
            color=color,
        )
    axis.set_xticks(positions, results["Model"], rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Held-out Test Precision / Recall / F1")
    axis.legend()
    figure.tight_layout()
    figure.savefig(add_plot_suffix("model_metrics", evaluation_id), dpi=150)
    plt.close(figure)

    save_confusion_plot(
        data["y_test"],
        data["gru_predictions"],
        data["labels"],
        "GRU Confusion Matrix",
        add_plot_suffix("gru_confusion_matrix", evaluation_id),
    )
    save_confusion_plot(
        data["y_test"],
        data["gnn_predictions"],
        data["labels"],
        "GNN Confusion Matrix",
        add_plot_suffix("gnn_confusion_matrix", evaluation_id),
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    normal = data["y_test"] == data["label_encoder"].transform(["NORMAL"])[0]
    axis.hist(ae_errors[normal], bins=10, alpha=0.7, label="Actual normal")
    axis.hist(ae_errors[~normal], bins=10, alpha=0.7, label="Actual fault")
    axis.axvline(
        ae_threshold,
        color="red",
        linestyle="--",
        label=f"Threshold = {ae_threshold:.4g}",
    )
    axis.set_xlabel("Reconstruction error (MSE)")
    axis.set_ylabel("Sample count")
    axis.set_title("Autoencoder Reconstruction Error")
    axis.legend()
    figure.tight_layout()
    figure.savefig(add_plot_suffix("autoencoder_errors", evaluation_id), dpi=150)
    plt.close(figure)


def append_results(results):
    output_path = RESULTS_DIR / "ml_results.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing_columns = list(pd.read_csv(output_path, nrows=0).columns)
        if existing_columns != RESULT_COLUMNS:
            raise ValueError(
                f"Existing {output_path} has unexpected columns; preserving it "
                "instead of replacing it."
            )
        results.to_csv(output_path, mode="a", header=False, index=False)
    else:
        results.to_csv(output_path, index=False)
    return output_path


def main():
    expected_models = [
        SAVED_DIR / "gru_fault_detector.keras",
        SAVED_DIR / "autoencoder_fault_detector.keras",
        SAVED_DIR / "autoencoder_metadata.json",
        SAVED_DIR / "gnn_fault_detector.pt",
    ]
    missing_models = [str(path) for path in expected_models if not path.exists()]
    if missing_models:
        raise FileNotFoundError(
            "Trained model files are missing. Run `python -m models.train_models` "
            "first. Missing: " + ", ".join(missing_models)
        )

    data = prepare_data()
    evaluation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label_ids = np.arange(len(data["labels"]))
    sample_count = len(data["y_test"])

    statistical_predictions, statistical_time = evaluate_statistical(data)

    gru_model = tf.keras.models.load_model(SAVED_DIR / "gru_fault_detector.keras")
    gru_predictions, gru_time = timed_ms_per_sample(
        lambda: predict_gru(gru_model, data["X_test"], data["y_test"]),
        sample_count,
    )

    autoencoder = tf.keras.models.load_model(
        SAVED_DIR / "autoencoder_fault_detector.keras"
    )
    metadata = json.loads(
        (SAVED_DIR / "autoencoder_metadata.json").read_text(encoding="utf-8")
    )
    ae_threshold = float(metadata["reconstruction_threshold"])
    ae_errors, ae_time = timed_ms_per_sample(
        lambda: reconstruction_errors(autoencoder, data["X_test"]),
        sample_count,
    )
    ae_predictions = ae_errors > ae_threshold
    actual_anomalies = data["y_test"] != data["label_encoder"].transform(["NORMAL"])[0]
    ae_metrics = classification_metrics(
        actual_anomalies.astype(int),
        ae_predictions.astype(int),
        labels=[0, 1],
    )

    gnn_model = load_gnn(SAVED_DIR / "gnn_fault_detector.pt")
    gnn_predictions, gnn_time = timed_ms_per_sample(
        lambda: predict_gnn(gnn_model, data["X_test"]),
        sample_count,
    )
    data["gru_predictions"] = gru_predictions
    data["gnn_predictions"] = gnn_predictions

    model_outputs = {
        "Statistical Threshold": (
            "3-class, chronological holdout",
            statistical_predictions,
            statistical_time,
        ),
        "GRU": ("3-class, chronological holdout", gru_predictions, gru_time),
        "GNN": ("3-class, chronological holdout", gnn_predictions, gnn_time),
    }
    rows = []
    for model_name in ("Statistical Threshold", "GRU"):
        task, predictions, detection_time = model_outputs[model_name]
        metrics = classification_metrics(
            data["y_test"], predictions, labels=label_ids
        )
        rows.append(
            {
                "Evaluation ID": evaluation_id,
                "Model": model_name,
                "Task": task,
                **metrics,
                "Detection Time (ms/sample)": detection_time,
                "Test Samples": sample_count,
            }
        )
        print(f"\n{model_name} (3-class)")
        print(pd.DataFrame(
            confusion_matrix(data["y_test"], predictions, labels=label_ids),
            index=[f"Actual {name}" for name in data["labels"]],
            columns=[f"Predicted {name}" for name in data["labels"]],
        ).to_string())

    rows.append(
        {
            "Evaluation ID": evaluation_id,
            "Model": "Autoencoder",
            "Task": "Normal vs anomaly, chronological holdout",
            **ae_metrics,
            "Detection Time (ms/sample)": ae_time,
            "Test Samples": sample_count,
        }
    )
    print("\nAutoencoder (binary: normal vs anomaly)")
    print(
        pd.DataFrame(
            confusion_matrix(
                actual_anomalies.astype(int), ae_predictions.astype(int), labels=[0, 1]
            ),
            index=["Actual normal", "Actual anomaly"],
            columns=["Predicted normal", "Predicted anomaly"],
        ).to_string()
    )
    print(f"Reconstruction threshold: {ae_threshold:.6f}")

    model_name = "GNN"
    task, predictions, detection_time = model_outputs[model_name]
    metrics = classification_metrics(data["y_test"], predictions, labels=label_ids)
    rows.append(
        {
            "Evaluation ID": evaluation_id,
            "Model": model_name,
            "Task": task,
            **metrics,
            "Detection Time (ms/sample)": detection_time,
            "Test Samples": sample_count,
        }
    )
    print(f"\n{model_name} (3-class)")
    print(
        pd.DataFrame(
            confusion_matrix(data["y_test"], predictions, labels=label_ids),
            index=[f"Actual {name}" for name in data["labels"]],
            columns=[f"Predicted {name}" for name in data["labels"]],
        ).to_string()
    )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    print("\nHeld-out test comparison (Autoencoder reports binary anomaly metrics):")
    print(results.drop(columns=["Evaluation ID"]).to_string(index=False))
    saved_results = append_results(results)
    save_comparison_plots(results, evaluation_id, data, ae_errors, ae_threshold)
    print(f"\nResults appended to: {saved_results}")
    print(f"Plots saved under: {PLOTS_DIR}")
    print(
        "Evaluation limitation: this small, stratified test split is preliminary "
        "and does not demonstrate generalization to unseen network conditions."
    )


if __name__ == "__main__":
    main()
