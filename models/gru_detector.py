from pathlib import Path

import numpy as np
import tensorflow as tf

from models.preprocess import RANDOM_SEED, make_sequences


def build_gru(input_shape, class_count):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.GRU(16),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(12, activation="relu"),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ],
        name="gru_fault_detector",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_gru(data: dict, model_path: Path):
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    X_train, y_train = make_sequences(data["X_train"], data["y_train"])
    X_val, y_val = make_sequences(data["X_val"], data["y_val"])

    model = build_gru(X_train.shape[1:], len(data["labels"]))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=8,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model, history


def predict_gru(model, features, labels):
    sequences, _ = make_sequences(features, labels)
    probabilities = model.predict(sequences, verbose=0)
    return np.argmax(probabilities, axis=1)
