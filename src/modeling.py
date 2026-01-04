from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def r2_loss(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return ss_res / (ss_tot + tf.keras.backend.epsilon())


def r2_metric(y_true, y_pred):
    return 1.0 - r2_loss(y_true, y_pred)


def build_lstm_model(
    window_size: int,
    num_features: int,
    horizon: int = config.HORIZON,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(window_size, num_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.1),
            layers.LSTM(32),
            layers.Dense(32, activation="relu"),
            layers.Dense(horizon),
        ]
    )
    model.compile(
        loss=r2_loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["mae", "mse", r2_metric],
    )
    return model


def train_model(
    model: keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    patience: int = 3,
):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,  # keep temporal order
        callbacks=callbacks,
        verbose=1,
    )
    return history
