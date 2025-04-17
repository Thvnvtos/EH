'''
strategies.py

This module will contain the classes for different real-time prediction strategies.
The class should contain a load_models() and predict() method.
A base strategy class, and subclasses for specific strategies (inheritance) could be used to properly organize this module.

'''


import tensorflow as tf
import numpy as np
import os, pickle
import akida





class Hierarchic_2Class:
    """
    A Class for the hierarchic strategy using six models for each couple of classes

    Methods
    -------
    load_models:
        load needed models and CSP objects from disk
    predict:
        use models to predict direction for input EEG epoch


    Parameters
    ----------
    model_type : str
        The used model type, can be 'akida' or 'cpu'
    using_chip : bool
        Flag for using the akida chip or not, could be useful when testing on CPU.
    """



    def __init__(self, model_type, using_chip):
        
        self.model_type = model_type
        self.using_chip = using_chip

        # All the patterns used for model and CSP objects naming
        self.models_naming = ['LR', 'UD', 'LU', 'LD', 'RU', 'RD']

        # Dictionnaries that will hold our models and CSP objects, self.models_naming will be used as keys
        self.models_dict = {}
        self.csp_dict = {}


    def load_models(self):
        """
        Load needed models and CSP objects from disk, and store them in self.models_dict and self.csp_dict.

        """
        
        # For both model types:
        # We simply iterate over all models and load them from disk with their respective CSP objective
        # And store everything in the dictionnaries 
        if self.model_type == 'cpu':
            print("================================= \n\n Using CPU Mode ...  \n")
            for naming in self.models_naming:
                loaded_model = tf.keras.models.load_model(os.path.join('saved_models', naming + '_cpu.tf'))
                with open(os.path.join('saved_models', 'csp' + naming + '_cpu.pkl'), 'rb') as f:
                    loaded_csp = pickle.load(f)

                self.models_dict[naming] = loaded_model
                self.csp_dict[naming] = loaded_csp
            print("=> All models successfully loaded \n")


        elif self.model_type == 'akida':
            print("================================= \n\n Using Akida Neuromorphic Mode ...  \n")

            # if using chip, we use akida.device() to access the chip, if only 1 chip is used it can be accessed with indice 0
            if self.using_chip:
                chip = akida.devices()[0]
                print(f"=> Akida chip detected = {chip} \n")
            else:
                print(f"=> Using Akida models without Chip \n")

            for naming in self.models_naming:
                loaded_model = akida.Model(os.path.join('saved_models', naming+'.fbz'))
                with open(os.path.join('saved_models', 'csp' + naming + '.pkl'), 'rb') as f:
                    loaded_csp = pickle.load(f)

                self.models_dict[naming] = loaded_model
                self.csp_dict[naming] = loaded_csp
            print("=> All models successfully loaded \n")

            # We use self.map() to map an akida model to a detected chip (device)
            # hw_only = True parameter forces the model to be mapped a single sequence on the Akida chip
            # if False, the akida model can be mapped as multiple sequences, which has less constraints (see README)
            if self.using_chip:
                for naming in self.models_naming:
                    self.models_dict[naming].map(chip, hw_only = True)

            if self.using_chip:
                print("=> All models successfully mapped to Akida Chip \n")
        
        else:
            print("Undefined Model Type ...")
            exit()

        


    def predict(self, X):
        """
        Given input EEG epoch X, predict using this strategy and return the direction 

        Parameters
        ----------
        X : np.ndarray
            input EEG epoch

        Returns
        -------
        direction : str
            predicted direction using the strategy, direction in ['left', 'right', 'up', 'down']
            
        """

        # We need 3 copies in total for this strategy
        # copies are necessary to make, since CSP will modify the original X
        X_copy1 = X.copy()
        X_final = X.copy()


        # CSP transform and expand dimension for LR and UD models
        # We don't transform X_final, since we don't know yet which model/CSP to use
        X_LR = self.csp_dict['LR'].transform(X)
        X_LR = np.expand_dims(X_LR, 3)

        X_UD = self.csp_dict['UD'].transform(X_copy1)
        X_UD = np.expand_dims(X_UD, 3)



        
        if self.model_type == 'akida':

            # Quantize to uint8 ([0, 255]) data linearly (see README for more details about this)
            X_LR = ((X_LR - X_LR.min()) / (X_LR.max() - X_LR.min()) * 255).astype(np.uint8)
            X_UD = ((X_UD - X_UD.min()) / (X_UD.max() - X_UD.min()) * 255).astype(np.uint8)

            # Zero-padding for the csp_components dimension to be compatible with akida's kernel_size constraints (see README for more details about this)
            X_LR = np.pad(X_LR, ((0, 0), (2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)
            X_UD = np.pad(X_UD, ((0, 0), (2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)

            # Use LR and UD model to get the 2 predictions
            out_LR = np.argmax(self.models_dict['LR'].forward(X_LR))
            out_UD = np.argmax(self.models_dict['UD'].forward(X_UD))
        
        elif self.model_type == 'cpu':
            # Use LR and UD model to get the 2 predictions
            out_LR = np.argmax(self.models_dict['LR'](X_LR))
            out_UD = np.argmax(self.models_dict['UD'](X_UD))




        # The line of code below gives the next model/CSP name to use based on the base models predictions, using string concatenation and multiplication properties of python
        # Example: 
        # out_LR = 0 (left)  and out_UD = 1 (down)
        # 'L' * 1 + 'R' * 0 + 'U' * 0 + 'D' * 1  = 'LD'
        prediction = 'L'*(1 - out_LR) + 'R'*(out_LR)  + 'U'*(1 - out_UD) + 'D'*(out_UD)



        # We use the prediction from previous models to do the final and 3rd prediction
        X_final = self.csp_dict[prediction].transform(X_final)
        X_final = np.expand_dims(X_final, 3)

        if self.model_type == 'akida':
            X_final = ((X_final - X_final.min()) / (X_final.max() - X_final.min()) * 255).astype(np.uint8)
            X_final = np.pad(X_final, ((0, 0), (2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)

            pred = self.models_dict[prediction].predict(X_final)                
            out = np.argmax(pred)
        
        elif self.model_type == 'cpu':
            out = np.argmax(self.models_dict[prediction](X_final))


        # Decide which prediction, transform it to a word and return final direction 
        direction = prediction[0]*(1 - out) + prediction[1]*(out)
        if direction == 'U': direction = 'up'
        elif direction == 'D': direction = 'down'
        elif direction == 'L': direction = 'left'
        elif direction == 'R': direction = 'right'

        return direction