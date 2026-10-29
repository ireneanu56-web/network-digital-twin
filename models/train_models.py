from pathlib import Path

import joblib

from models.autoencoder_detector import train_autoencoder
from models.gnn_detector import train_gnn
from models.gru_detector import train_gru
from models.preprocess import prepare_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_DIR = PROJECT_ROOT / "models" / "saved"


def main():
    data = prepare_data()
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "imputer": data["imputer"],
            "scaler": data["scaler"],
            "label_encoder": data["label_encoder"],
            "feature_columns": data["feature_columns"],
            "sequence_length": data["sequence_length"],
        },
        SAVED_DIR / "preprocessing.joblib",
    )
    print(
        f"Dataset split: {len(data['y_train'])} train, "
        f"{len(data['y_val'])} validation, {len(data['y_test'])} test samples."
    )
    print(
        "Note: the GRU uses short windows from the small held-out splits; "
        "windows never cross split boundaries."
    )
    print("\n[1/3] Training GRU...")
    train_gru(data, SAVED_DIR / "gru_fault_detector.keras")

    print("\n[2/3] Training Autoencoder...")
    _, threshold, _, _ = train_autoencoder(
        data,
        SAVED_DIR / "autoencoder_fault_detector.keras",
        SAVED_DIR / "autoencoder_metadata.json",
    )
    print(f"Normal-only reconstruction threshold: {threshold:.6f}")

    print("\n[3/3] Training GNN...")
    train_gnn(data, SAVED_DIR / "gnn_fault_detector.pt")
    print("\nAll three model training runs completed and model files were saved.")
    print(
        "GNN limitation: the source CSV contains network-wide samples, not "
        "per-link/node telemetry; the same observed sample is used as graph context."
    )


if __name__ == "__main__":
    main()
