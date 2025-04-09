import mne, os
from typing import Tuple
import numpy as np

import utils_old
from config_BCICIV2a import Config



def subject_dataset(config: Config, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:

    gdf = read_GDF_subject(subject_id, config.data_path)
    gdf = select_EEG_channels(gdf, config.used_channels, config.channels_dict)


    X_0, X_1 = get_rawEEG_arrays(gdf, config.used_classes, config.t_start, config.t_end)    
    y_0, y_1 = np.ones(X_0.shape[0]), np.zeros(X_1.shape[0])



    # X shape = (N, channels, time), y shape = (N)
    X, y = np.concatenate((X_0, X_1)), np.concatenate((y_0, y_1))

    X = filter_rawEEG(X, config.lowcut, config.highcut)

    return X, y





def read_GDF_subject(subject_id: int, data_path: str) -> mne.io.Raw:
    # Reads GDF file for given subject from data_path and returns an mne RawGDF

    gdf_path = os.path.join(data_path, f'A0{subject_id}T.gdf')
    gdf = mne.io.read_raw_gdf(gdf_path, eog=['EOG-left', 'EOG-central', 'EOG-right'])
    gdf.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

    return gdf




def select_EEG_channels(raw_gdf:mne.io.Raw, used_channels: list[int], channels_dict: dict) -> mne.io.Raw:
    # removes non selected channels, and returns new RawGDF with only selected channels

    raw_gdf.pick_channels([channels_dict[ch] for ch  in used_channels ])

    return raw_gdf



def get_rawEEG_arrays(raw_gdf:mne.io.Raw,  used_classes: list[str], t_start: float, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
    # Return EEG raw data np arrays for both selected classes from t_start to t_end
    # np arrays shape = (N, Channels, Time)


    #Get events from annotations in raw_gdf
    events = mne.events_from_annotations(raw_gdf)

    # From BCICIV2a data description and annotations of gdf file
    event_dict={
        'reject':1,
        'eye move':2,
        'eye open':3,
        'eye close':4,
        'new run':5,
        'new trial':6,
        'Left':7,
        'Right':8,
        'Foot':9,
        'Tongue':10,
    }

    class0_id = event_dict[used_classes[0]]
    class1_id = event_dict[used_classes[1]]

    #Create epochs sliced on tmin and tmax from the time of the different events, here the onset of the arrow corresponding to the class
    epochs = mne.Epochs(raw_gdf, events[0], event_id=[class0_id, class1_id], tmin = t_start, tmax = t_end)

    # use string(id) to select from epochs
    return epochs[str(class0_id)].get_data(), epochs[str(class1_id)].get_data()




def filter_rawEEG(raw_eeg: np.ndarray, lowcut: float, highcut: float) -> np.ndarray:
    # Use bandpass and notch filtering, as suggested by ConsciousLabs

    eeg = utils_old.bandpass_filter(raw_eeg, lowcut, highcut, sr=250)
    filtered_eeg = utils_old.notch_filter(eeg, sr=250)

    return filtered_eeg
  


