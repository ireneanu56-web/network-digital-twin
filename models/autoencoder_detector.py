import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.preprocess import RANDOM_SEED


def build_autoencoder(feature_count):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(feature_count,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(2, activation="relu", name="latent"),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(feature_count, activation="linear"),
        ],
        name="telemetry_autoencoder",
    )


def reconstruction_errors(model, features):
    reconstructed = model.predict(features, verbose=0)
    return np.mean(np.square(features - reconstructed), axis=1)


def train_autoencoder(data: dict, model_path: Path, metadata_path: Path):
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    normal_class = data["label_encoder"].transform(["NORMAL"])[0]
    normal_train = data["X_train"][data["y_train"] == normal_class]
    normal_val = data["X_val"][data["y_val"] == normal_class]
    if len(normal_train) < 2 or len(normal_val) < 1:
        raise ValueError(
            "The normal training/validation split is too small to fit an autoencoder"
        )

    model = build_autoencoder(data["X_train"].shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        normal_train,
        normal_train,
        validation_data=(normal_val, normal_val),
        epochs=100,
        batch_size=8,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    validation_errors = reconstruction_errors(model, normal_val)
    threshold = float(np.quantile(validation_errors, 0.95))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "reconstruction_threshold": threshold,
                "threshold_method": "95th percentile of normal validation errors",
                "normal_validation_samples": int(len(normal_val)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return model, threshold, validation_errors, history
