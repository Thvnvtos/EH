class Config:
    
    ################################################
    #            General configuration             #
    ################################################


    random_seed = 0
    data_path = '../EEG_RAW_DATA_NEW'                                             


    # Choose from: 'CLeft', 'CRight', 'CUp' and 'CDown'
    used_classes = [['CLeft'], ['CRight']]

    # Session type = 'S2' or 'S3'
    session_type = 'S2'

    test_session = 9

    t_start = 0.0
    t_end = 3.0

    window_size = 2.0
    stride = 0.1


    lowcut = 0.5#7
    highcut = 35

    # select from  ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4']

    used_channels = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4']


    
    # Filtering parameters:
    n_csp_components = 3


    model_type = '2D'

    epochs = 200
    batch_size = 32
    n_splits = 5
    val_size = 0.15

    learning_rate = 2e-4
    label_smoothing = 0.05
    weight_decay = 1e-5
