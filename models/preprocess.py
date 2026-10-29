from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from monitoring.fault_detector import CSV_PATH, FAULT_MAP


FEATURE_COLUMNS = [
    "packet_loss",
    "avg_rtt",
    "jitter",
    "throughput_mbps",
    "retransmissions",
]
LABELS = ["NORMAL", "LATENCY_FAULT", "PACKET_LOSS_FAULT"]
RANDOM_SEED = 42
SEQUENCE_LENGTH = 3


def prepare_data(csv_path: Path = CSV_PATH) -> dict:
    """Clean telemetry and produce reproducible, stratified, scaled splits."""
    data = pd.read_csv(csv_path)
    required = {"fault_type", *FEATURE_COLUMNS}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )
    if data.empty:
        raise ValueError(f"No telemetry rows found in {csv_path}")

    data = data.copy()
    data["actual_label"] = (
        data["fault_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(FAULT_MAP)
    )
    unknown_labels = data.loc[data["actual_label"].isna(), "fault_type"].unique()
    if len(unknown_labels):
        raise ValueError(f"Unknown fault_type labels in dataset: {unknown_labels.tolist()}")
    if "timestamp" in data:
        data["_sort_timestamp"] = pd.to_datetime(
            data["timestamp"], errors="coerce", utc=True
        )
        data = data.sort_values("_sort_timestamp", kind="stable").drop(
            columns="_sort_timestamp"
        )
        data = data.reset_index(drop=True)
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    if data[FEATURE_COLUMNS].isna().all(axis=1).any():
        invalid_rows = int(data[FEATURE_COLUMNS].isna().all(axis=1).sum())
        raise ValueError(f"{invalid_rows} dataset rows contain no usable features")

    label_encoder = LabelEncoder()
    label_encoder.fit(LABELS)
    targets = label_encoder.transform(data["actual_label"])
    train_indices = []
    validation_indices = []
    test_indices = []
    for label in np.unique(targets):
        class_indices = np.flatnonzero(targets == label)
        test_size = max(1, round(len(class_indices) * 0.20))
        validation_size = max(1, round(len(class_indices) * 0.20))
        if len(class_indices) <= test_size + validation_size:
            raise ValueError(
                f"Not enough samples in class {label} for three data splits"
            )
        train_end = len(class_indices) - test_size - validation_size
        validation_end = len(class_indices) - test_size
        train_indices.extend(class_indices[:train_end])
        validation_indices.extend(class_indices[train_end:validation_end])
        test_indices.extend(class_indices[validation_end:])
    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = np.asarray(validation_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    train_indices.sort()
    validation_indices.sort()
    test_indices.sort()

    raw_features = data[FEATURE_COLUMNS]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(raw_features.iloc[train_indices]))
    X_val = scaler.transform(imputer.transform(raw_features.iloc[validation_indices]))
    X_test = scaler.transform(imputer.transform(raw_features.iloc[test_indices]))

    y_train = targets[train_indices]
    y_val = targets[validation_indices]
    y_test = targets[test_indices]

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "y_val": y_val.astype(np.int64),
        "y_test": y_test.astype(np.int64),
        "scaler": scaler,
        "imputer": imputer,
        "label_encoder": label_encoder,
        "feature_columns": FEATURE_COLUMNS,
        "labels": list(label_encoder.classes_),
        "raw_data": data,
        "raw_train": data.iloc[train_indices].copy(),
        "raw_val": data.iloc[validation_indices].copy(),
        "raw_test": data.iloc[test_indices].copy(),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "sequence_length": SEQUENCE_LENGTH,
    }


def make_sequences(features, labels, sequence_length=SEQUENCE_LENGTH):
    """Build short windows inside a split, padding only with observed rows."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows")
    if len(features) == 0:
        raise ValueError("Cannot create sequences from an empty split")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    sequences = np.empty(
        (len(features), sequence_length, features.shape[1]), dtype=np.float32
    )
    for endpoint in range(len(features)):
        start = max(0, endpoint - sequence_length + 1)
        window = features[start : endpoint + 1]
        if len(window) < sequence_length:
            padding = np.repeat(window[:1], sequence_length - len(window), axis=0)
            window = np.concatenate((padding, window), axis=0)
        sequences[endpoint] = window
    return sequences, labels
