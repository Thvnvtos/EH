import os, random 
import numpy as np
import pandas as pd
#import torch

from scipy import signal


def read_eeg_csv(data_path, subject_id):
    '''
    Returns pandas dataframe of given subject, removing first 2 lines containing experiment information
    '''
    file_name = ''

    pattern = f'N{subject_id:02d}'
    for s in os.listdir(data_path):
        if pattern in s:
            file_name = s

    csv_path = os.path.join(data_path, file_name)
    return pd.read_csv(csv_path, skiprows=2, delimiter=';')




def get_slices(raw_eeg_dfs, config):

    '''
    
        Returns a list slices of len n_subjects, containing the raw eeg data and class + rep informations

        slice[i][0] : (total number of slices, n_channels/electrodes, time) array of i-th subject
        slice[i][1] : list of len = total number of slices containing (class_id, rep_id) of i-th subject
    
    '''



    slices = []

    for df in raw_eeg_dfs:

        subject_X= []
        subject_y = []

        for class_name in config.classes:
        
            pattern_start = f'Motor Imagery Start:{config.session}:{class_name}'
            pattern_end = f'Motor Imagery End:{config.session}:{class_name}'

            event_start_ts = df[df['Marker']==pattern_start].index                          # All event start timesteps for all events done in the experiment for the subject/class
            

            for i,t_event in enumerate(event_start_ts):

                slice_start_ts = config.t_start
                slice_end_ts =  slice_start_ts + config.window_size          

                while slice_end_ts <= config.t_end + 1e-9:

                    slice_start_ts_sr = int(config.sr * slice_start_ts) + t_event
                    slice_end_ts_sr = int(config.sr * slice_end_ts) + t_event


                    raw_eeg_slice = np.array(df.loc[slice_start_ts_sr : slice_end_ts_sr - 1, config.used_channels]).transpose(1, 0)

                    raw_eeg_slice = bandpass_filter(raw_eeg_slice, config.lowcut, config.highcut, config.sr)
                    raw_eeg_slice = notch_filter(raw_eeg_slice, config.sr)


                    # to avoid size mismatch, slice by window size 
                    subject_X.append(raw_eeg_slice[:, :int(config.sr * config.window_size)])
                    #subject_X.append(raw_eeg_slice)
                    subject_y.append((config.class_name_id[class_name], i))

                    slice_start_ts += config.window_stride
                    slice_end_ts += config.window_stride



        slices.append((np.array(subject_X), subject_y))


    return slices





def bandpass_filter(data, lowcut, highcut, sr, order=4):
    '''
    Requires input data to be in shape (-1, -1, time) or (-1, time)
    '''
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data



def notch_filter(data, sr, freqs = [(48,52), (98,102)], q=30):
    '''
    Requires input data to be in shape (-1, -1, time) or (-1, time)
    '''
    for freq in freqs:
        f0 = freq[0] + (freq[1] - freq[0]) / 2  
        w0 = f0 / (sr / 2) 
        b, a = signal.iirnotch(w0, q)
        
        # Apply the filter along the time axis (axis=-1) 
        data = signal.filtfilt(b, a, data, axis=-1)
    
    return data





###
# Not Used currently, might use if I switch to PyTorch
###

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