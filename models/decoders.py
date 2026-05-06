#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras import Model, Input, callbacks, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout,
    MultiHeadAttention, LayerNormalization, Embedding,
    Conv1D, Conv2D, AveragePooling2D, Flatten, Add,
    SeparableConv2D, DepthwiseConv2D,
    BatchNormalization, Activation, SpatialDropout2D
)
from tensorflow.keras.constraints import max_norm

class Decoder(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, **kwargs):
        raise NotImplementedError

class LinReg(Decoder):
    def __init__(self, fit_intercept=True, **kwargs):
        self.model = LinearRegression(fit_intercept=fit_intercept, **kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X)

class RandForest(Decoder):
    def __init__(
        self,
        n_estimators=300,
        criterion="squared_error",
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
        **kwargs,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X)

class FNN(Decoder):
    """
    Feed-forward neural network regressor.
    """
    def __init__(
        self,
        hidden1=64,
        hidden2=32,
        hidden3=32,
        dropout1=0.20,
        dropout2=0.15,
        dropout3=0.10,
        lr=1e-3,
        epochs=50,
        batch_size=32,
        checkpoint_path="best_ffn.keras",
        early_stopping=True,
        patience=5,
        min_delta=0.0
    ):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.model = None
        self.input_neurons = None

    def _build_model(self):
        model = Sequential(
            [
                Input(shape=(self.input_neurons,), name="input"),
                Dense(self.hidden1, activation="relu", name="hidden1"),
                Dropout(self.dropout1, name="dropout1"),
                Dense(self.hidden2, activation="relu", name="hidden2"),
                Dropout(self.dropout2, name="dropout2"),
                Dense(self.hidden3, activation="relu", name="hidden3"),
                Dropout(self.dropout3, name="dropout3"),
                Dense(1, name="out")
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss="mean_squared_error",
        )
        return model

    def fit(self, X, y, X_val=None, y_val=None, verbose=1, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        input_neurons = int(X.shape[1])
        if self.model is None or self.input_neurons != input_neurons:
            self.input_neurons = input_neurons
            self.model = self._build_model()

        callbacks_list = []
        validation_data = None

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            validation_data = (X_val, y_val)

            callbacks_list.append(
                callbacks.ModelCheckpoint(
                    filepath=self.checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=verbose,
                )
            )

            if self.early_stopping:
                callbacks_list.append(
                    callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=self.patience,
                        min_delta=self.min_delta,
                        restore_best_weights=True,
                        verbose=verbose,
                    )
                )

        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=verbose,
            **kwargs,
        )

        if any(isinstance(cb, callbacks.ModelCheckpoint) for cb in callbacks_list):
            self.model = tf.keras.models.load_model(self.checkpoint_path)
        return self

    def predict(self, X, verbose=1, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y_pred = self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=verbose
        )
        return y_pred.ravel()

class RNN(Decoder):
    def __init__(
        self,
        lstm1_units=128,
        lstm2_units=64,
        dense1_units=64,
        dense2_units=32,
        dropout1=0.1,
        dropout2=0.1,
        lr=1e-3,
        epochs=25,
        batch_size=32,
        checkpoint_path="best_lstm.keras",
        early_stopping=True,
        patience=5,
        min_delta=0.0
    ):
        self.lstm1_units = lstm1_units
        self.lstm2_units = lstm2_units
        self.dense1_units = dense1_units
        self.dense2_units = dense2_units
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.seq_len = None
        self.n_channels = None
        self.model = None

    def _compile(self, lr):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mean_squared_error"
        )

    def _build_model(self):
        inp = Input(shape=(self.seq_len, self.n_channels), name="input")
        x = LSTM(self.lstm1_units, return_sequences=True, name="lstm_1")(inp)
        x = LayerNormalization(name="ln_1")(x)
        x = LSTM(self.lstm2_units, name="lstm_2")(x)
        x = LayerNormalization(name="ln_2")(x)
        x = Dense(self.dense1_units, activation="relu", name="dense_1")(x)
        x = Dropout(self.dropout1, name="dropout_1")(x)
        x = Dense(self.dense2_units, activation="relu", name="dense_2")(x)
        x = Dropout(self.dropout2, name="dropout_2")(x)
        out = Dense(1, name="out")(x)
        model = Model(inputs=inp, outputs=out)

        self.model = model
        self._compile(self.lr)
        return model

    def fit(self, X, y, X_val=None, y_val=None, verbose=1, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        n_channels = int(X.shape[2])
        seq_len = int(X.shape[1])
        if self.model is None or self.n_channels != n_channels or self.seq_len != seq_len:
            self.n_channels = n_channels
            self.seq_len = seq_len
            self.model = self._build_model()

        callbacks_list = []
        validation_data = None

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            validation_data = (X_val, y_val)

            callbacks_list.append(
                callbacks.ModelCheckpoint(
                    filepath=self.checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=verbose
                )
            )

            if self.early_stopping:
                callbacks_list.append(
                    callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=self.patience,
                        min_delta=self.min_delta,
                        restore_best_weights=True,
                        verbose=verbose
                    )
                )

        self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=verbose,
            **kwargs
        )

        if any(isinstance(cb, callbacks.ModelCheckpoint) for cb in callbacks_list):
            self.model = tf.keras.models.load_model(self.checkpoint_path)
        return self

    def fine_tune(
        self,
        X_new,
        y_new,
        X_val=None,
        y_val=None,
        lr=None,
        epochs=None,
        freeze_lstm=True,
        checkpoint_path=None,
        early_stopping=None,
        patience=None,
        min_delta=None,
        verbose=1,
        **kwargs
    ):
        """
        Fine-tune on new data by freezing LSTM layers and training the dense head.
        """
        X_new = np.asarray(X_new, dtype=np.float32)
        y_new = np.asarray(y_new, dtype=np.float32).reshape(-1, 1)

        if lr is None:
            lr = self.lr
        if epochs is None:
            epochs = self.epochs
        if early_stopping is None:
            early_stopping = self.early_stopping
        if patience is None:
            patience = self.patience
        if min_delta is None:
            min_delta = self.min_delta

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path.replace(".keras", "_finetune.keras")

        # Freeze/unfreeze layers
        if freeze_lstm:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.LSTM):
                    layer.trainable = False
        else:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.LSTM):
                    layer.trainable = True

        # (Dense layers remain trainable by default; ensure they are)
        for name in ("dense_1", "dense_2", "out"):
            layer = self.model.get_layer(name=name)
            layer.trainable = True

        # Recompile after changing trainable flags
        self._compile(lr)

        # Fine tune model
        callbacks_list = []
        validation_data = None

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            validation_data = (X_val, y_val)

            callbacks_list.append(
                callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=verbose
                )
            )

            if early_stopping:
                callbacks_list.append(
                    callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=patience,
                        min_delta=min_delta,
                        restore_best_weights=True,
                        verbose=verbose
                    )
                )

        self.model.fit(
            X_new, y_new,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=verbose,
            **kwargs
        )

        if any(isinstance(cb, callbacks.ModelCheckpoint) for cb in callbacks_list):
            self.model = tf.keras.models.load_model(checkpoint_path)
        return self

    def predict(self, X, verbose=1, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y_pred = self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=verbose
        )
        return y_pred.ravel()

class Transformer(Decoder):
    """
    Encoder-only Transformer regressor.
    """
    def __init__(
        self,
        d_model=128,
        num_heads=4,
        ff_dim=64,
        dropout_encoder=0.1,
        num_encoder_layers=2,
        conv_filters=32,
        conv_kernel_size=3,
        dense1_units=16,
        dense2_units=8,
        dropout1=0.1,
        dropout2=0.1,
        lr=1e-3,
        epochs=25,
        batch_size=32,
        checkpoint_path="best_transformer.keras",
        early_stopping=True,
        patience=5,
        min_delta=0.0
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_encoder = dropout_encoder
        self.num_encoder_layers = num_encoder_layers
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.dense1_units = dense1_units
        self.dense2_units = dense2_units
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.seq_len = None
        self.n_channels = None
        self.model = None

    def _compile(self, lr):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mean_squared_error"
        )

    def _encoder_block(self, x, block_id):
        # Multi-head self-attention (bidirectional, no causal mask)
        attn = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            name=f"mha_{block_id}"
        )(x, x)

        attn = Dropout(self.dropout_encoder, name=f"attn_dropout_{block_id}")(attn)
        x = Add(name=f"attn_skip_{block_id}")([x, attn])
        x = LayerNormalization(epsilon=1e-6, name=f"attn_ln_{block_id}")(x)

        # Feed-forward network
        ffn = Dense(self.ff_dim, activation="relu", name=f"ffn_dense_1_{block_id}")(x)
        ffn = Dense(self.d_model, name=f"ffn_dense_2_{block_id}")(ffn)
        ffn = Dropout(self.dropout_encoder, name=f"ffn_dropout_{block_id}")(ffn)
        x = Add(name=f"ffn_skip_{block_id}")([x, ffn])
        x = LayerNormalization(epsilon=1e-6, name=f"ffn_ln_{block_id}")(x)
        return x

    def _build_model(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        inp = Input(shape=(self.seq_len, self.n_channels), name="input")

        # Project tokens (EEG snapshots) into d_model
        x = Dense(self.d_model, name="token_projection")(inp)

        # Learned positional embeddings
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        positions = tf.expand_dims(positions, axis=0)  # (1, seq_len)
        pos = Embedding(input_dim=self.seq_len, output_dim=self.d_model, name="pos_embedding")(positions)

        x = Add(name="add_positional")([x, pos])  # Broadcast add (1, seq_len, d_model)

        # Encoder stack
        for i in range(self.num_encoder_layers):
            x = self._encoder_block(x, block_id=i)

        # 1D conv over the sequence (context-aware features)
        x = Conv1D(
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            strides=1,
            padding="same",
            activation="relu",
            name="conv1d"
        )(x)

        x = Flatten(name="flatten")(x)

        # Regression head
        x = Dense(self.dense1_units, activation="relu", name="dense_1")(x)
        x = Dropout(self.dropout1, name="dropout_1")(x)
        x = Dense(self.dense2_units, activation="relu", name="dense_2")(x)
        x = Dropout(self.dropout2, name="dropout_2")(x)
        out = Dense(1, name="out")(x)

        model = Model(inputs=inp, outputs=out)
        self.model = model
        self._compile(self.lr)
        return model

    def fit(self, X, y, X_val=None, y_val=None, verbose=1, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        n_channels = int(X.shape[2])
        seq_len = int(X.shape[1])

        # Lazy build / rebuild if shape changed
        if self.model is None or self.n_channels != n_channels or self.seq_len != seq_len:
            self.n_channels = n_channels
            self.seq_len = seq_len
            self.model = self._build_model()

        callbacks_list = []
        validation_data = None

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            validation_data = (X_val, y_val)

            callbacks_list.append(
                callbacks.ModelCheckpoint(
                    filepath=self.checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=verbose
                )
            )

            if self.early_stopping:
                callbacks_list.append(
                    callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=self.patience,
                        min_delta=self.min_delta,
                        restore_best_weights=True,
                        verbose=verbose
                    )
                )

        self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=verbose,
            **kwargs
        )
        if any(isinstance(cb, callbacks.ModelCheckpoint) for cb in callbacks_list):
            self.model = tf.keras.models.load_model(self.checkpoint_path)
        return self

    def predict(self, X, verbose=1, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y_pred = self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=verbose
        )
        return y_pred.ravel()

class EEGNetRegressor(Decoder):
    """
    EEGNet-style regressor adapted from https://github.com/vlawhern/arl-eegmodels.

    Expects X with shape:
        (n_trials, seq_len, n_channels)

    Internally reshapes to:
        (n_trials, n_channels, seq_len, 1)
    """
    def __init__(
        self,
        dropoutRate=0.5, # within-subject
        kernLength=5, # adapted, smaller since our epochs are 200ms long at 100Hz
        F1=8,
        D=2,
        F2=16,
        norm_rate=0.25,
        dropoutType="Dropout",
        pool1=4, 
        pool2=2, # smaller to have valid pooling with 200ms epochs
        lr=1e-3,
        epochs=50,
        batch_size=32,
        checkpoint_path="best_eegnet.keras",
        early_stopping=True,
        patience=5,
        min_delta=0.0
    ):
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        self.dropoutType = dropoutType
        self.pool1 = pool1
        self.pool2 = pool2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.seq_len = None
        self.n_channels = None
        self.model = None

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                "EEGNetRegressor expects X with shape "
                "(n_trials, seq_len, n_channels)."
            )

        # From (trials, time, channels) to (trials, channels, time, 1)
        X = np.transpose(X, (0, 2, 1))
        X = X[..., np.newaxis]
        return X

    def _compile(self, lr):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mean_squared_error"
        )

    def _build_model(self):
        if self.dropoutType == "SpatialDropout2D":
            dropout_layer = SpatialDropout2D
        elif self.dropoutType == "Dropout":
            dropout_layer = Dropout
        else:
            raise ValueError(
                "dropoutType must be either 'SpatialDropout2D' or 'Dropout'."
            )

        inp = Input(shape=(self.n_channels, self.seq_len, 1), name="input")

        x = Conv2D(
            self.F1,
            (1, self.kernLength),
            padding="same",
            use_bias=False,
            name="temporal_conv"
        )(inp)

        x = BatchNormalization(name="bn_1")(x)

        x = DepthwiseConv2D(
            (self.n_channels, 1),
            use_bias=False,
            depth_multiplier=self.D,
            depthwise_constraint=max_norm(1.0),
            name="spatial_depthwise_conv"
        )(x)

        x = BatchNormalization(name="bn_2")(x)
        x = Activation("elu", name="elu_1")(x)
        x = AveragePooling2D((1, self.pool1), name="avg_pool_1")(x)
        x = dropout_layer(self.dropoutRate, name="dropout_1")(x)

        x = SeparableConv2D(
            self.F2,
            (1, 16),
            use_bias=False,
            padding="same",
            name="separable_conv"
        )(x)

        x = BatchNormalization(name="bn_3")(x)
        x = Activation("elu", name="elu_2")(x)
        x = AveragePooling2D((1, self.pool2), name="avg_pool_2")(x)
        x = dropout_layer(self.dropoutRate, name="dropout_2")(x)

        x = Flatten(name="flatten")(x)

        out = Dense(
            1,
            name="out",
            kernel_constraint=max_norm(self.norm_rate)
        )(x)

        model = Model(inputs=inp, outputs=out)
        self.model = model
        self._compile(self.lr)
        return model

    def fit(self, X, y, X_val=None, y_val=None, verbose=1, **kwargs):
        X_raw = np.asarray(X, dtype=np.float32)

        if X_raw.ndim != 3:
            raise ValueError(
                "EEGNetRegressor expects X with shape "
                "(n_trials, seq_len, n_channels)."
            )

        seq_len = int(X_raw.shape[1])
        n_channels = int(X_raw.shape[2])

        if (
            self.model is None
            or self.seq_len != seq_len
            or self.n_channels != n_channels
        ):
            self.seq_len = seq_len
            self.n_channels = n_channels
            self.model = self._build_model()

        X = self._prepare_X(X_raw)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        callbacks_list = []
        validation_data = None

        if X_val is not None and y_val is not None:
            X_val = self._prepare_X(X_val)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
            validation_data = (X_val, y_val)

            callbacks_list.append(
                callbacks.ModelCheckpoint(
                    filepath=self.checkpoint_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=verbose
                )
            )

            if self.early_stopping:
                callbacks_list.append(
                    callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=self.patience,
                        min_delta=self.min_delta,
                        restore_best_weights=True,
                        verbose=verbose
                    )
                )

        self.model.fit(
            X,
            y,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=verbose,
            **kwargs
        )

        if any(isinstance(cb, callbacks.ModelCheckpoint) for cb in callbacks_list):
            self.model = tf.keras.models.load_model(self.checkpoint_path)

        return self

    def predict(self, X, verbose=1, **kwargs):
        X = self._prepare_X(X)

        y_pred = self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=verbose
        )

        return y_pred.ravel()
