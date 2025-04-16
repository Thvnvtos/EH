"""
utils.py

This module is mainly used now for the bandpass and notch filter.
I kept it separate because some functions here could still be useful for the old EEG dataset.



"""


import os 
import numpy as np
import pandas as pd

from scipy import signal






def bandpass_filter(data, lowcut, highcut, sr, order=4):
    """
    Applies a Butterworth bandpass filter to raw EEG data.

    Parameters
    ----------
    data : np.ndarray
        Input signal array. Expected shape is either (channels, time) or (N, channels, time).
    lowcut : float
        Lower cutoff frequency of the bandpass filter in Hz.
    highcut : float
        Upper cutoff frequency of the bandpass filter in Hz.
    sr : float
        Sampling rate of the signal in Hz.
    order : int, optional
        Order of the Butterworth filter. Default is 4.

    Returns
    -------
    np.ndarray
        Bandpass-filtered data with the same shape as the input.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data




def notch_filter(data, sr, freqs=[(48, 52), (98, 102)], q=30):
    """
    Applies multiple IIR notch filters to remove specific frequency bands from the EEG signal.

    Parameters
    ----------
    data : np.ndarray
        Input signal array. Expected shape is either (channels, time) or (N, channels, time).
    sr : float
        Sampling rate of the signal in Hz.
    freqs : list of tuple of float, optional
        List of frequency bands (as (low, high)) to remove from the signal. Each band defines a notch range.
        For example, [(48, 52), (98, 102)] targets 50 Hz and 100 Hz noise.
    q : float, optional
        Quality factor of the notch filter. Higher values mean narrower notches. Default is 30.

    Returns
    -------
    np.ndarray
        Notch-filtered signal with the same shape as the input.
    """
    for freq in freqs:
        f0 = freq[0] + (freq[1] - freq[0]) / 2  
        w0 = f0 / (sr / 2) 
        b, a = signal.iirnotch(w0, q)
        data = signal.filtfilt(b, a, data, axis=-1)
    
    return data






########################################################################


# Below are functions used on first recorded dataset with 33 subjects .


########################################################################








def read_eeg_csv(data_path, subject_id):
    """
    (Used for the 1st Airbus EEG dataset)
    Reads EEG data from a CSV file for a given subject as a pandas dataframe, skipping the first 2 metadata lines.

    Parameters
    ----------
    data_path : str
        Path to the directory containing EEG CSV files.
    subject_id : int
        The ID number of the subject from 1 to 33 (subjects 14 and 18 files do not contain any data) .

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the EEG data for the specified subject.
    """


    file_name = ''

    # format subject id to string pattern like N01, N23, etc ...
    pattern = f'N{subject_id:02d}'
    
    for s in os.listdir(data_path):
        if pattern in s:
            file_name = s

    csv_path = os.path.join(data_path, file_name)
    return pd.read_csv(csv_path, skiprows=2, delimiter=';')










def get_slices(config, raw_eeg_dfs):
    """
    Extracts and filters EEG time-window slices from raw EEG DataFrames for each subject.

    Parameters
    ----------
    raw_eeg_dfs : list of pandas.DataFrame
        A list of DataFrames, one per subject, each containing EEG recordings with time-indexed rows.
        Each DataFrame must include a 'Marker' column and columns for EEG channels.
    config : object
        Main configuration file (see config.py)

    Returns
    -------
    list of tuple slices
        A list of length n_subjects, where each element is a tuple:
        - slices[i][0] : np.ndarray of shape (n_slices, n_channels, timesteps) for i-th subject
        - slices[i][1] : list of length n_slices containing tuples (class_id, repetition_id)
    """

    slices = []  # Final list containing all slices
    
    for df in raw_eeg_dfs:
        subject_X = []  # Holds EEG data slices for one subject
        subject_y = []  # Holds (class_id, repetition_id) labels for one subject

        for class_name in config.classes:
            # Construct marker strings to identify start of motor imagery tasks
            pattern_start = f'Motor Imagery Start:{config.session}:{class_name}'
            

            # Locate the time indices (row indices) where motor imagery starts
            event_start_ts = df[df['Marker'] == pattern_start].index

            for i, t_event in enumerate(event_start_ts):
                # Set initial window boundaries relative to event
                slice_start_ts = config.t_start
                slice_end_ts = slice_start_ts + config.window_size

                # Slide window until it passes t_end
                while slice_end_ts <= config.t_end + 1e-9:
                    # Convert start and end times from seconds to indices using the sampling rate
                    slice_start_ts_sr = int(config.sr * slice_start_ts) + t_event
                    slice_end_ts_sr = int(config.sr * slice_end_ts) + t_event

                    # Extract EEG slice and transpose to (channels, time)
                    raw_eeg_slice = np.array(
                        df.loc[slice_start_ts_sr: slice_end_ts_sr - 1, config.used_channels]
                    ).transpose(1, 0)

                    # Apply bandpass and notch filters
                    raw_eeg_slice = bandpass_filter(raw_eeg_slice, config.lowcut, config.highcut, config.sr)
                    raw_eeg_slice = notch_filter(raw_eeg_slice, config.sr)

                    # Trim to exact window size to ensure consistency
                    subject_X.append(raw_eeg_slice[:, :int(config.sr * config.window_size)])

                    # Append the label (class_id, repetition_id)
                    subject_y.append((config.class_name_id[class_name], i))

                    # Move the window forward by the stride
                    slice_start_ts += config.window_stride
                    slice_end_ts += config.window_stride

        # Aggregate all slices and labels for the subject
        slices.append((np.array(subject_X), subject_y))

    return slices













#########################

# This was used to set different random seeds, especially when using PyTorch, NOT USED NOW

'''

def set_seed_torch(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This flag only allows cudnn algorithms that are determinestic unlike .benchmark
    torch.backends.cudnn.deterministic = True

    #this flag enables cudnn for some operations such as conv layers and RNNs, 
    # which can yield a significant speedup.
    torch.backends.cudnn.enabled = False

    # This flag enables the cudnn auto-tuner that finds the best algorithm to use
    # for a particular configuration. (this mode is good whenever input sizes do not vary)
    torch.backends.cudnn.benchmark = False

    # I don't know if this is useful, look it up.
    #os.environ['PYTHONHASHSEED'] = str(seed)

'''