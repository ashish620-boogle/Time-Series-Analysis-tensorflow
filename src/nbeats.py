from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import saving

from .modeling import r2_loss, r2_metric
from tensorflow.keras import layers

from . import config


@saving.register_keras_serializable(package="Custom", name="NBeatsBlock")
class NBeatsBlock(layers.Layer):
    """Simplified N-BEATS fully-connected block."""

    def __init__(self, units: int, backcast_len: int, forecast_len: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.fc1 = layers.Dense(units, activation="relu")
        self.fc2 = layers.Dense(units, activation="relu")
        self.fc3 = layers.Dense(units, activation="relu")
        self.fc4 = layers.Dense(units, activation="relu")
        self.theta = layers.Dense(backcast_len + forecast_len, activation="linear")

    def call(self, inputs):
        x = layers.Flatten()(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        theta = self.theta(x)
        backcast = theta[..., : self.backcast_len]
        forecast = theta[..., self.backcast_len :]
        return backcast, forecast


def build_nbeats_model(
    window_size: int,
    horizon: int = config.HORIZON,
    num_features: int = 1,
    units: int = 256,
    stacks: int = 4,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """
    Build an N-BEATS model for univariate/multivariate windowed inputs.
    """
    input_layer = layers.Input(shape=(window_size, num_features), name="input_layer")
    residuals = input_layer
    forecast_sum = None

    backcast_len = window_size * num_features
    forecast_len = horizon  # predicting scalar close price

    for _ in range(stacks):
        backcast, block_forecast = NBeatsBlock(
            units=units,
            backcast_len=backcast_len,
            forecast_len=forecast_len,
        )(residuals)

        backcast_reshaped = layers.Reshape((window_size, num_features))(backcast)
        residuals = layers.subtract([residuals, backcast_reshaped])

        block_forecast = layers.Reshape((forecast_len,))(block_forecast)
        if forecast_sum is None:
            forecast_sum = block_forecast
        else:
            forecast_sum = layers.add([forecast_sum, block_forecast])

    forecast = layers.Lambda(lambda x: x[:, -horizon:], name="forecast")(forecast_sum)
    model = keras.Model(inputs=input_layer, outputs=forecast, name="nbeats_model")
    model.compile(
        loss=r2_loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["mae", "mse", r2_metric],
    )
    return model
