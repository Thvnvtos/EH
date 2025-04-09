class Config:
    
    ################################################
    #            General configuration             #
    ################################################


    random_seed = 0
    data_path = '..\BCICIV_2a_gdf' #'../BCICIV_2a_gdf'                                                    


    # Choose from: 'Left', 'Right', 'Foot' and 'Tongue'
    used_classes = ['Left', 'Right']


    t_start = -0.1
    t_end = 3.0


    lowcut = 0.5#7
    highcut = 35

    # select from 1 to 22 below
    used_channels = [   2,3,4,5,6,
                        7,8,9,10,11,12,13,
                        14,15,16,17,18
                     ]#[i for i in range(1, 23)]

    # for electrode id and placement, check dataset description in: https://www.bbci.de/competition/iv/    
    channels_dict = {
    1: 'EEG-Fz',
    2: 'EEG-0',
    3: 'EEG-1',
    4: 'EEG-2',
    5: 'EEG-3',
    6: 'EEG-4',
    7: 'EEG-5',
    8: 'EEG-C3',
    9: 'EEG-6',
    10: 'EEG-Cz',
    11: 'EEG-7',
    12: 'EEG-C4',
    13: 'EEG-8',
    14: 'EEG-9',
    15: 'EEG-10',
    16: 'EEG-11',
    17: 'EEG-12',
    18: 'EEG-13',
    19: 'EEG-14',
    20: 'EEG-Pz',
    21: 'EEG-15',
    22: 'EEG-16'
    }

    
    # Filtering parameters:
    n_csp_components = 3