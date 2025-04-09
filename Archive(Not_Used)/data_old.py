import numpy as np
from torch.utils.data import Dataset
from mne.decoding import CSP

import torch

import os, sys

import utils_old





class DataHandler:


    def __init__(self, config, device):

        self.config = config
        self.device = device

        print("\n\n=========> Initializing DataHandler...\n")
        
        print("=> Reading CSV files...", sep=' => ')
        raw_eeg_dfs = []
        print("=> Training subject list:  ", config.subjects)
        for subject_id in config.subjects:
            raw_eeg_dfs.append(utils_old.read_eeg_csv(config.data_path, subject_id))
        print("Ok!\n")

        print("=> Slicing data...", sep=' => ')
        self.slices = utils_old.get_slices(raw_eeg_dfs, config)
        print("Ok!\n")
    







    def __split__(self, fold_seed, subjects, n_split):
        print("\n ==> Splitting data... \n")
        

        # reassign ids after removing some subjects
        subjects = [i for i in range(len(subjects))]

        X_train_raw_bySujbect, X_test_raw_bySujbect = [], []
        y_train_bySubject, y_test_bySubject = [], []


        class_count_per_subject = []

        for subject_id in subjects:
            
            X, y  = self.slices[subject_id]
            y = np.array(y)

            # Calculate repitions count for each class for current subject (to be used in fold splitting)
            class_rep_count = []
            for class_label in range(self.config.n_classes):
                class_rep_count.append(max([rep for c, rep in y if c==class_label]))


            # data_by_rep contains for each rep: (index_in_original_data_array, class)
            data_by_rep = {}
            for i, (class_label, rep) in enumerate(y):
                if rep not in data_by_rep:
                    data_by_rep[rep] = []
                data_by_rep[rep].append((i, class_label)) 

                
            train_indexes = []
            test_indexes = []

            print(f"=> Subject_index {subject_id}: ")

            class_count = [0, 0]
            for class_label in range(self.config.n_classes):

                utils_old.set_seed_torch(fold_seed)

                reps_list = np.arange(class_rep_count[class_label]+1)
                np.random.shuffle(reps_list)
                

                split_index = int(n_split * len(reps_list))

                train_reps = reps_list[:split_index]
                test_reps = reps_list[split_index:]

                
                print(f"Class{class_label} - Train reps = ", train_reps)
                print(f"Class{class_label} - Test reps  = ", test_reps)

                for rep in train_reps:
                    for elem in data_by_rep[rep]:
                        if elem[1] == class_label:
                            train_indexes.append(elem[0])
                            class_count[class_label] += 1 
                
                for rep in test_reps:
                    for elem in data_by_rep[rep]:
                        if elem[1] == class_label:
                            test_indexes.append(elem[0])

            class_count_per_subject.append(class_count)

            X_train = X[train_indexes]
            y_train = np.array([class_label for class_label, _ in y[train_indexes]])

            X_test = X[test_indexes]
            y_test = np.array([class_label for class_label, _ in y[test_indexes]])

            print(f"\nTrain size = {X_train.shape}")
            print(f"Test size  = {X_test.shape}\n")

            X_train_raw_bySujbect.append(X_train)
            X_test_raw_bySujbect.append(X_test)

            y_train_bySubject.append(y_train)
            y_test_bySubject.append(y_test)
        
        return X_train_raw_bySujbect, y_train_bySubject, X_test_raw_bySujbect, y_test_bySubject, class_count_per_subject









    def get_DataSubjectWiseCSP(self, X_train_raw_bySujbect, y_train_bySubject, X_test_raw_bySujbect, y_test_bySubject, transform_into='csp_space'):
        
        X_train, X_test = [], []
        y_train, y_test = [], []

        for subject_id in range(len(y_train_bySubject)):        

            utils_old.set_seed_torch(0)
            
            # Redirecting stdout to avoid seeing messages from uneeded csp transformation
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            # Calculate CSP using the training data for this subject
            csp = CSP(n_components = self.config.n_csp_components ,reg=None, log=None, norm_trace=False, transform_into=transform_into)
            X_train_subject = csp.fit_transform(X_train_raw_bySujbect[subject_id], y_train_bySubject[subject_id])

            # Use the CSP from train on test data.
            X_test_subject = csp.transform(X_test_raw_bySujbect[subject_id])

            sys.stdout.close()
            sys.stdout = old_stdout


            if transform_into == 'csp_space':
                X_train.append(np.expand_dims(X_train_subject, axis=1).astype(np.float32))
                X_test.append(np.expand_dims(X_test_subject, axis=1).astype(np.float32))
            else:
                X_train.append(X_train_subject.astype(np.float32))
                X_test.append(X_test_subject.astype(np.float32))


            y_train.append(y_train_bySubject[subject_id])
            y_test.append(y_test_bySubject[subject_id])

        return X_train, y_train, X_test, y_test







    
class GenericDataset(Dataset):

    def __init__(self, config, X, y):

        self.config = config

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]