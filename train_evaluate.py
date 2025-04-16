# TensorFlow / Keras imports
from tensorflow import keras
import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras.callbacks import ModelCheckpoint

# Standard libraries
import sys, os
import numpy as np
import matplotlib.pyplot as plt

# CSP from MNE for spatial filtering
from mne.decoding import CSP

# Local dataset handling module
import dataset




def train_evalute(config, X_train, Y_train, X_valid, Y_valid, model, choose_on='val_acc', csp=None, plot_graphs=True, train=True):
    """
    Train and evaluate models.

    Parameters
    ----------
    config : object
        Main configuration file (see config.py)
    X_*, Y_* : np.ndarray
        Training/Validation EEG data and labels
    model : tf.keras.Model
        Keras model instance to train and evaluate.
    choose_on : str, optional
        Metric to monitor when saving best model (default is 'val_acc').
    csp : mne.decoding.CSP or None
        Optional CSP instance; if None, a new one will be created and fitted.
    plot_graphs : bool, optional
        Whether to plot training curves (default is True).
    train : bool, optional
        If False, disables the `ModelCheckpoint` callback.

    Returns
    -------
    model : tf.keras.Model
        Trained Keras model loaded with best weights.
    csp : mne.decoding.CSP
        CSP object used on data.
    best_val_acc : float
        Best validation accuracy reached during training.
    """

    # Slice EEG epochs from original rep
    X_train, Y_train = dataset.slice_EEG_epoch(config, X_train, Y_train)
    X_valid, Y_valid = dataset.slice_EEG_epoch(config, X_valid, Y_valid)

    # Temporarily suppress CSP logs and stdout
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Apply CSP (if not provided, fit it here)
    if csp is None:
        csp = CSP(
            n_components=config.n_csp_components,
            reg=None,
            log=None,
            norm_trace=False,
            transform_into='csp_space'
        )
        X_train = csp.fit_transform(X_train, Y_train)
    else:
        X_train = csp.transform(X_train)

    X_valid = csp.transform(X_valid)

    # Re-enable logging
    sys.stdout.close()
    sys.stdout = old_stdout

    # # Expand dims for CNN2D (N, H, W, C)
    X_train = np.expand_dims(X_train, axis=3)   # X_train shape = (N, CSP_components, Time, 1)
    X_valid = np.expand_dims(X_valid, axis=3)

    # Zero-padding for the csp_components dimension to be compatible with akida's kernel_size constraints (see README for more details about this)
    if config.model_type == 'akida':
        X_train = np.pad(X_train, ((0, 0), (2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)
        X_valid = np.pad(X_valid, ((0, 0), (2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)

    # Convert labels to one-hot format
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=len(config.used_classes))
    Y_valid = tf.keras.utils.to_categorical(Y_valid, num_classes=len(config.used_classes))

    # Compile the model
    model.compile(
        loss=CategoricalCrossentropy(from_logits=True, label_smoothing=config.label_smoothing),
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate, weight_decay=config.weight_decay),
        metrics=[keras.metrics.CategoricalAccuracy(name='acc')]
    )

    # Train the model, using checkpoint if training mode is enabled
    if train:
        checkpoint_cb = ModelCheckpoint(
            os.path.join('saved_models', 'best_model_fold.tf'),
            save_best_only=True,
            monitor=choose_on,
            mode='max',
            verbose=0,
            save_format='tf'
        )
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_valid, Y_valid),
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=1,
            callbacks=[checkpoint_cb]
        )
    else:
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_valid, Y_valid),
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=1
        )

    # Record best validation accuracy
    best_val_acc = max(history.history['val_acc'])

    # Optionally plot training curves
    if plot_graphs:
        plt.figure(figsize=(15, 5))
        plt.title(
            f"| N_examples = {X_train.shape[0]} / {X_valid.shape[0]} | Best_val_acc={100 * best_val_acc:.2f}% "
        )

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(["Train Loss", "Val Loss"])

        plt.subplot(1, 2, 2)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.legend(["Train Acc", "Val Acc"])

        plt.show(block=False)
        plt.pause(0.001)

    # Reload best model from checkpoint
    model = tf.keras.models.load_model(os.path.join('saved_models', 'best_model_fold.tf'))

    return model, csp, best_val_acc