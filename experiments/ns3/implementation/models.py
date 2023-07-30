"""Tensorflow models.

Important! This file uses the following python syntax a lot:
function(*, arg1, arg2)

If you are not familiar with it: all arguments after the `*` are keyword-only
arguments; i.e. they _must_ be provided with a keyword (but have no defaults).

This is to ensure that they are not passed accidentially using a list of
arguments. There are many arguments and it would be easy to mix up the
order, thus we use this protection.
"""
# pylint: disable=import-outside-toplevel

import numpy as np
import tensorflow as tf
from experiment_helpers.data import Path
from sklearn.model_selection import train_test_split

from ..config import Ns3ExperimentConfig


def save_model(model, path):
    """Save to model."""
    model.save_weights(Path(path) / 'weights')

# Workload Prediction.
# ====================


def load_workload_model(path, config=None):
    """Load the model."""
    config = Ns3ExperimentConfig.with_updates(config)
    model = compile_workload_model(
        n_workloads=config.workloads,
        horizon=config.horizon,
        hidden_layers=config.workload_hidden_layers,
        hidden_units=config.workload_hidden_units,
        learning_rate=config.workload_learning_rate,
    )
    model.load_weights(Path(path) / 'weights').expect_partial()
    return model


def compile_workload_model(*,  # All arguments are keyword-only to avoid errors.
                           n_workloads: int,
                           horizon: int,
                           hidden_layers: int,
                           hidden_units: int,
                           learning_rate: float):
    """Create a new model for workload prediction."""
    n_inputs = horizon

    # Input layer.
    layers = [
        tf.keras.layers.Dense(hidden_units, input_shape=(n_inputs,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ]
    # Further hidden layers.
    for _ in range(hidden_layers - 1):  # type: ignore
        layers += [
            tf.keras.layers.Dense(hidden_units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]
    # Output layer.
    layers += [tf.keras.layers.Dense(n_workloads)]

    model = tf.keras.models.Sequential(layers)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'],
                  )
    return model


def train_workload_model(x_train, y_train, config=None):
    """Train a model and return it."""
    config = Ns3ExperimentConfig.with_updates(config)
    model = compile_workload_model(
        n_workloads=config.workloads,
        learning_rate=config.workload_learning_rate,
        horizon=config.horizon,
        hidden_layers=config.workload_hidden_layers,
        hidden_units=config.workload_hidden_units,
    )
    callbacks = []

    if config.workload_patience > 0:
        assert config.workload_validation_split > 0
        # Early stopping callback.
        stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.workload_patience,
            restore_best_weights=True
        )
        callbacks.append(stop)

    # Progress callback.
    # progress = tfa.callbacks.TQDMProgressBar(
    #     show_overall_progress=(verbose >= 1),
    #     show_epoch_progress=(verbose >= 2),
    # )
    # callbacks.append(progress)

    # Train
    if config.workload_validation_split > 0:
        # Use sklearn to split off validation data, keras can't do it easily.
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            test_size=config.workload_validation_split,
            shuffle=True,
            stratify=y_train,
            random_state=config.random_state,
        )
        validation_data = (x_val, y_val)
    else:
        validation_data = None  # type: ignore

    model.fit(x_train, y_train,
              validation_data=validation_data,
              epochs=config.workload_epochs,
              batch_size=config.workload_learning_batchsize,
              callbacks=callbacks, verbose=0)

    return model


def predict_workload(model, x_in):
    """Add a softmax layer to the workload model and predict probabilities."""
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    return probability_model.predict(x_in)


# Transmit Time Prediction.
# =========================

@tf.function
def neg_log_likelihood(y, prediction):
    """Negative log-likelihood."""
    # pylint: disable=invalid-name
    return -logscore(y, prediction)


@tf.function
def logscore(y, prediction):
    """Logscore, which is the log-likelihood."""
    # pylint: disable=invalid-name
    mean, logsigma = tf.unstack(prediction, axis=-1)
    mse = -.5 * tf.square((mean - y) / tf.exp(logsigma))
    return mse - logsigma - 0.5 * tf.math.log(2 * tf.constant(np.pi))


@tf.function
def rmse(y, prediction):
    """Root mean square error of predicted gauss.

    High if mean is off or uncertainty is high.

    Note that if X ~ N(mu, sigma):
    X - y ~ N(mu - y, sigma)
    E[(X -y)**2] = (mu-y)**2 + sigma**2
    """
    # pylint: disable=invalid-name
    mu, logsigma = tf.unstack(prediction, axis=-1)
    mse = tf.square(mu - y) + tf.square(tf.exp(logsigma))
    return tf.sqrt(mse)


def load_transtime_model(path, config=None):
    """Load the model."""
    config = Ns3ExperimentConfig.with_updates(config)
    model = compile_transtime_model(
        horizon=config.horizon,
        hidden_layers=config.transtime_hidden_layers,
        hidden_units=config.transtime_hidden_units,
        learning_rate=config.transtime_learning_rate,
    )
    model.load_weights(Path(path) / 'weights').expect_partial()
    return model


def compile_transtime_model(*,  # All arguments are keyword-only.
                            horizon: int,
                            hidden_layers: int,
                            hidden_units: int,
                            learning_rate: float):
    """Create a new model for transmit time prediction."""
    n_inputs = 2 * horizon - 1  # type: ignore

    # Input layer.
    layers = [
        tf.keras.layers.Dense(hidden_units, input_shape=(n_inputs,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    ]
    # Further hidden layers.
    for _ in range(hidden_layers - 1):  # type: ignore
        layers += [
            tf.keras.layers.Dense(hidden_units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ]
    # Output layer.
    layers += [tf.keras.layers.Dense(1)]
    model = tf.keras.models.Sequential(layers)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()],
                  )
    return model


def train_transtime_model(x_train, y_train, config=None):
    """Train a model and return it."""
    config = Ns3ExperimentConfig.with_updates(config)
    model = compile_transtime_model(
        horizon=config.horizon,
        hidden_layers=config.transtime_hidden_layers,
        hidden_units=config.transtime_hidden_units,
        learning_rate=config.transtime_learning_rate,
    )
    callbacks = []

    if config.transtime_patience > 0:
        assert config.transtime_validation_split > 0
        # Early stopping callback.
        stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.transtime_patience, restore_best_weights=True
        )
        callbacks.append(stop)

    # Progress callback.
    # progress = tfa.callbacks.TQDMProgressBar(
    #     show_overall_progress=(verbose >= 1),
    #     show_epoch_progress=(verbose >= 2),
    # )
    # callbacks.append(progress)

    # Train
    if config.transtime_validation_split > 0:
        # Use sklearn to split off validation data, keras can't do it easily.
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            test_size=config.transtime_validation_split, shuffle=True,
            random_state=config.random_state,
        )
        validation_data = (x_val, y_val)
    else:
        validation_data = None  # type: ignore

    model.fit(x_train, y_train,
              validation_data=validation_data,
              epochs=config.transtime_epochs,
              batch_size=config.transtime_learning_batchsize,
              callbacks=callbacks, verbose=0)

    return model
