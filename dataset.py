"""
dataset.py

This module contains the necessary functions to make numpy array datasets (X,y) for any combination of classes:
Example: 
[[C1], [C2]] : 2 way classification C1 VS C2
[[C1, C2], [C3, C4]]: 2 way classification C1 and C2 VS C3 and C4
[[C1], [C2, C3, C4]]: 2 way classification C1 VS C2, C3 and C4
[[C1], [C2], [C3], [C4]]: 4 way classification

With functionnality make slices using sliding window

"""


import os
from typing import Tuple
import numpy as np
import pandas as pd

# Local imports
import utils
from config import Config


def session_dataset(config: Config, session_ID: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses EEG data for a specific session ID.

    Parameters
    ----------
    config : Config
        Main configuration file (see config.py)
    session_ID : str
        Identifier for the EEG recording session.

    Returns
    -------
    X : np.ndarray
        Array of filtered raw EEG data with shape (N, Channels, Time).
    y : np.ndarray
        Corresponding class labels for each trial with shape (N, ).
    """
    # Convert start and end times from seconds to sample indices using the sampling rate
    t_start = int(config.t_start * config.sampling_rate)
    t_end = int(config.t_end * config.sampling_rate)

    # Prepare a list for each class to store corresponding EEG segments
    # class could be a group of different mouvements
    X = [[] for _ in range(len(config.used_classes))]

    # Iterate through all files in the dataset directory
    for f in sorted(os.listdir(config.data_path)):
        for i, classes in enumerate(config.used_classes):
            for class_id in classes:
                # Check if file corresponds to a specific class, session type, and session ID
                if class_id in f and config.session_type in f and session_ID in f:
                    df = pd.read_csv(os.path.join(config.data_path, f), delimiter=',')
                    df = df.loc[:, config.used_channels]  # Select used EEG channels
                    X[i].append(np.array(df)[t_start:t_end])  # Slice time window and add to class list

    # Create label array: assign label i to all samples in class i
    y = [[i] * len(X[i]) for i in range(len(X))]
    X, y = np.concatenate(X).transpose(0, 2, 1), np.concatenate(y)  # Transpose to (N, Channels, Time)

    # Apply bandpass and notch filtering
    X = filter_rawEEG(X, config.lowcut, config.highcut)

    return X, y







def slice_EEG_epoch(config: Config, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice EEG signals into overlapping windows (epochs) using a sliding window approach.

    Parameters
    ----------
    config : Config
        Main configuration file (see config.py)
    X : np.ndarray
        Input  EEG data with shape (N, Channels, Time).
    y : np.ndarray
        Corresponding class labels for each trial with shape (N, ).

    Returns
    -------
    X_sliced : np.ndarray
        Slices of EEG data (N_slices, Channels, Time(window_size) ).
    y_sliced : np.ndarray
        Labels corresponding to each slice (N_slices).
    """

    stride = int(config.stride * config.sampling_rate)  # Slide in number of samples
    window_size = int(config.window_size * config.sampling_rate)
    t_start = int(config.t_start * config.sampling_rate)
    t_end = int(config.t_end * config.sampling_rate)

    X_sliced, y_sliced = [], []

    # Apply sliding window to each trial
    for i in range(len(X)):
        for start in range(t_start, t_end - window_size + 1, stride):
            segment = X[i][:, start:start + window_size][np.newaxis]
            X_sliced.append(segment)
            y_sliced.append(y[i])

    # Combine all windowed segments into arrays
    X_sliced, y_sliced = np.concatenate(X_sliced), np.array(y_sliced)

    return X_sliced, y_sliced


def filter_rawEEG(raw_eeg: np.ndarray, lowcut: float, highcut: float) -> np.ndarray:
    """
    Apply preprocessing filters to raw EEG data: bandpass and notch filter.

    Parameters
    ----------
    raw_eeg : np.ndarray
        EEG data with shape (N, Channels, Time).
    lowcut : float
        Low cutoff frequency for bandpass filter.
    highcut : float
        High cutoff frequency for bandpass filter.

    Returns
    -------
    filtered_eeg : np.ndarray
        Filtered EEG data with shape (N, Channels, Time).
    """

    # Bandpass filter to retain only relevant frequency components
    eeg = utils.bandpass_filter(raw_eeg, lowcut, highcut, sr=250)

    # Notch filter to remove power line noise (e.g., 50/60 Hz)
    filtered_eeg = utils.notch_filter(eeg, sr=250)

    return filtered_eeg