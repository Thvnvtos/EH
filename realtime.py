import tensorflow as tf

import numpy as np
import os, pickle


from mne.decoding import CSP
from dataset import filter_rawEEG


from Serial_class import serial_class
import threading
import time
import sys, logging, warnings

import akida
import requests

from queue import Queue



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')







server_ip =  "http://192.168.69.53:8000"

real_data = False

# Select from ['akida' , 'cpu']
model_type = 'cpu'

models_naming = ['LR', 'UD', 'LU', 'LD', 'RU', 'RD']
models_dict = {}
csp_dict = {}



if model_type == 'cpu':
    print("================================= \n\n Using CPU Mode ...  \n")
    for naming in models_naming:
        
        loaded_model = tf.keras.models.load_model(os.path.join('saved_models', naming + '_cpu.tf'))
        with open(os.path.join('saved_models', 'csp' + naming + '_cpu.pkl'), 'rb') as f:
            loaded_csp = pickle.load(f)

        models_dict[naming] = loaded_model
        csp_dict[naming] = loaded_csp
    print("=> All models successfully loaded \n")

elif model_type == 'akida':

    print("================================= \n\n Using Akida Neuromorphic Mode ...  \n")

    chip = akida.devices()[0]

    print(f"=> Akida chip detected = {chip} \n")

    for naming in models_naming:
        loaded_model = akida.Model(os.path.join('saved_models', naming+'.fbz'))
        with open(os.path.join('saved_models', 'csp' + naming + '.pkl'), 'rb') as f:
            loaded_csp = pickle.load(f)

        models_dict[naming] = loaded_model
        csp_dict[naming] = loaded_csp
    
    print("=> All models successfully loaded \n")


    for naming in models_naming:
        models_dict[naming].map(chip, hw_only = True)

    print("=> All models successfully mapped to Akida Chip \n")


else:
    print("Undefined Model Type ...")
    exit()


time.sleep(2)



class serial_fake_data:
    def __init__(self):
        self.data = True
        self.data_queue = Queue()


        # Same size as a real EEG headset
        self.sample_line = np.random.rand(4,16)
        self.sample_ts = [0,1,2,3]

    def fill_queue(self):
        self.data_queue.put((self.sample_line,self.sample_ts))




if __name__ == '__main__':

    print("================================= \n\n EEG Streaming and Classification Starting ...  \n\n\n")
    time.sleep(1)


    #Gestion port com
    if real_data: 
        serial_flux = serial_class()
        serial_flux.init_port()

        thread_a = threading.Thread(target=serial_flux.reception, name='ta')
        thread_a.start()
    else:
        serial_flux = serial_fake_data()

    time.sleep(2)

    if not serial_flux.data:
        print('No serial flux , program is closing')
        serial_flux.terminate()
        sys.exit()


    EEGraw_stack = []

    try:
        while True:
            if not real_data:
                serial_flux.fill_queue()

            while not serial_flux.data_queue.empty():
                array_data_list , ts_data = serial_flux.data_queue.get()

                for i in range(len(array_data_list)):
                    ts_value = ts_data[i] + (i*4) # Obtient le timestamp associe
                    EEGraw_stack.append(array_data_list[i][:13])    


                if len(EEGraw_stack) > 500:
                    if model_type == 'akida':
                        last_epoch_raw = EEGraw_stack[-250:]
                    elif model_type == 'cpu':
                        last_epoch_raw = EEGraw_stack[-500:]
                    # Preprocess raw data
                    X = np.array(last_epoch_raw)
                    
                    X = X.transpose(1,0)                    
                    X = filter_rawEEG(X, 0.5, 35)
                    X = np.expand_dims(X, 0)

                    # Make 2 copies for different CSP and models
                    X_copy1 = X.copy()
                    X_final = X.copy()


                    # CSP on base models LR and UD
                    X_LR = csp_dict['LR'].transform(X)
                    X_LR = np.expand_dims(X_LR, 3)

                    X_UD = csp_dict['UD'].transform(X_copy1)
                    X_UD = np.expand_dims(X_UD, 3)



                    # Linear Quantization
                    if model_type == '2D':
                        X_LR = ((X_LR - X_LR.min()) / (X_LR.max() - X_LR.min()) * 255).astype(np.uint8)
                        X_UD = ((X_UD - X_UD.min()) / (X_UD.max() - X_UD.min()) * 255).astype(np.uint8)

                        X_LR = np.pad(X_LR, ((0, 0), (2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)
                        X_UD = np.pad(X_UD, ((0, 0), (2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)

                        out_LR = np.argmax(models_dict['LR'].forward(X_LR))
                        out_UD = np.argmax(models_dict['UD'].forward(X_UD))
                    
                    elif model_type == 'cpu':
                        out_LR = np.argmax(models_dict['LR'](X_LR))
                        out_UD = np.argmax(models_dict['UD'](X_UD))


                    prediction = 'L'*(1 - out_LR) + 'R'*(out_LR)  + 'U'*(1 - out_UD) + 'D'*(out_UD)


                    # Finalizing prediction:

                    X_final = csp_dict[prediction].transform(X_final)
                    X_final = np.expand_dims(X_final, 3)

                    if model_type == 'akida':
                        X_final = ((X_final - X_final.min()) / (X_final.max() - X_final.min()) * 255).astype(np.uint8)
                        X_final =np.pad(X_final, ((0, 0), (2, 2), (0, 0), (0, 0)), mode='constant', constant_values=0)

                        chip.soc.power_measurement_enabled = True
                        pred = models_dict[prediction].predict(X_final)
                        #print(models_dict[prediction].statistics)
                        
                        floor_power = chip.soc.power_meter.floor
                        #print(f'Floor power: {floor_power:.2f} mW  (Idle power consumption)')
                        
                        out = np.argmax(pred)
                              
                    
                    elif model_type == 'cpu':
                        out = np.argmax(models_dict[prediction](X_final))


                    direction = prediction[0]*(1 - out) + prediction[1]*(out)
                    if direction == 'U': direction = 'up'
                    elif direction == 'D': direction = 'down'
                    elif direction == 'L': direction = 'left'
                    elif direction == 'R': direction = 'right'

                    #command = 'cls' if os.name == 'nt' else 'clear'
                    #os.system(command)
                    
                    #print("===================================================================\n")
                    print(f"Final Prediction = {direction} | intermediary pred = {prediction}")
                    #print("\n===================================================================\n")
                    #requests.post(f"{server_ip}/push_direction", json={"direction": direction, "confidence": 1.0})



                    # Clean temp raw EEG stack 
                    if len(EEGraw_stack) > 1000:
                        EEGraw_stack[:] = last_epoch_raw

            
            time.sleep(0.002)  # Ajouter un delai pour eviter d'occuper 100 du CPU

    except KeyboardInterrupt:
        # Arreter le thread proprement lors d'une interruption (Ctrl + C)
        if real_data:
            serial_flux.terminate() # Fermeture de la connexion EEG
            thread_a.join()
