"""
config.py

This class is used to centralize all the parameters of an experiment for ease of changing the whole configuration.

"""



class Config:
    
    ################################################
    #            General configuration             #
    ################################################

    random_seed = 0
    data_path = '../EEG_RAW_DATA_NEW'                                             
    sampling_rate = 250

    # Choose from: 'CLeft', 'CRight', 'CUp' and 'CDown'
    # It's possible to choose any combination of classes such as:
    # [[C1], [C2]] : 2 way classification C1 VS C2
    # [[C1, C2], [C3, C4]]: 2 way classification C1 and C2 VS C3 and C4
    # [[C1], [C2, C3, C4]]: 2 way classification C1 VS C2, C3 and C4
    # [[C1], [C2], [C3], [C4]]: 4 way classification
    # Etc, ...
    used_classes = [['CLeft'], ['CRight']]

    # Session type = 'S2' or 'S3' 
    # S2 = Imagined Motor Imagery
    # S3 = Intuitive Motor Imagery
    session_type = 'S2'

    # Which recording session to use for testing, all others will be used for training
    # S2 = choose from [1, 11]
    # S3 = choose from [1, 15] 
    test_session = 9

    # Start and End relative to the event, which will be used  (THIS IS NOT THE TIME WINDOW USED TO FEED THE MODELS)
    t_start = 0.0
    t_end = 3.0


    # Window size and stride, used for slicing from the epoch above, window size should be smaller than t_end - t_start 
    # Choose 1.0 for model_type = 'akida' and 2.0 for model_type = 'cpu' (this can be changed in different ways, see README)
    window_size = 2.0
    stride = 0.1


    # bandpass filter parameters 
    lowcut = 0.5
    highcut = 35


    # Select from  ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4']
    used_channels = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4']


    
    # Number of csp components to use
    n_csp_components = 3


    # Select from ['akida' , 'cpu']
    model_type = 'akida'


    # Pre-training parameters

    epochs = 200
    batch_size = 32

    learning_rate = 1e-4
    label_smoothing = 0.05
    weight_decay = 1e-5