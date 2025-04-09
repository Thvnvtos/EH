import os
from typing import Tuple
import numpy as np
import pandas as pd


import utils
from config_NewEEG import Config






def session_dataset(config: Config, session_ID: str) -> Tuple[np.ndarray, np.ndarray]:

    t_start = int(config.t_start * 250)
    t_end = int(config.t_end * 250)
    

    X = [[] for i in range(len(config.used_classes))]

    for f in sorted(os.listdir(config.data_path)):

        for i, classes in enumerate(config.used_classes):
            for class_id in classes:
                
                if class_id in f and config.session_type in f and session_ID in f:
                    df = pd.read_csv(os.path.join(config.data_path, f), delimiter=',')
                    df = df.loc[:, config.used_channels]
                    X[i].append(np.array(df)[t_start:t_end])
                    
        
    

    y = [[i] * len(X[i]) for i in range(len(X))]
    X, y = np.concatenate(X).transpose(0, 2, 1), np.concatenate(y)


    X = filter_rawEEG(X, config.lowcut, config.highcut)

    return X, y





def slice_EEG_epoch(config: Config, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    stride = int(config.stride * 250)
    window_size = int(config.window_size * 250)
    t_start = int(config.t_start * 250)
    t_end = int(config.t_end * 250)


    X_sliced, y_sliced = [], []


    for i in range(len(X)):
        for start in range(t_start, t_end - window_size + 1, stride):
                X_sliced.append(X[i][:, start:start + window_size][np.newaxis])
                y_sliced.append(y[i])

    
    X_sliced, y_sliced = np.concatenate(X_sliced), np.array(y_sliced)

    return X_sliced, y_sliced








def filter_rawEEG(raw_eeg: np.ndarray, lowcut: float, highcut: float) -> np.ndarray:
    # Use bandpass and notch filtering, as suggested by ConsciousLabs

    eeg = utils.bandpass_filter(raw_eeg, lowcut, highcut, sr=250)
    filtered_eeg = utils.notch_filter(eeg, sr=250)

    return filtered_eeg
  


