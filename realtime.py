'''
realtime.py

The main script for running the real-time pipeline from reading EEG data, prediction on Akida and finally sending the predicted direction to the server

'''

import numpy as np
import os, threading, time, sys, logging, warnings

import requests
from queue import Queue

# Local imports
from Serial_class import serial_class
from strategies import Hierarchic_2Class
from dataset import filter_rawEEG




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')





# The POST request server ip address and port
server_ip =  "http://192.168.69.53:8000"

# You can use simulated fake data if set to False
using_real_data = False

# You can test using the saved Akida models on CPU
using_chip = False



# Select from ['akida' , 'cpu']
model_type = 'cpu'



strategy = Hierarchic_2Class(model_type, using_chip)
strategy.load_models()



time.sleep(3.0)



class serial_fake_data:
    """
    This class is used to simulate Serial_class.serial_class with random values
    it has a similar usage to the real class
    Simply read from data_queue, but also needs to be filled each time 4 rows are read from it

    """
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
    if using_real_data: 
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


    # This list will be used as a queue to store read EEG data
    # When it surpasses the minimum window size, we use it's content and we clean it once it's used to avoid memory errors
    # I prefered using a list instead of a queue for ease of conversion to numpy, just one line more is needed to convert it to queue (if needed for faster latency)
    EEGraw_queue = []
    minimum_EEG_size = 250 if model_type == 'akida' else 500

    try:
        while True:
            if not using_real_data:
                serial_flux.fill_queue()

            while not serial_flux.data_queue.empty():
                array_data_list , ts_data = serial_flux.data_queue.get()

                for i in range(len(array_data_list)):
                    ts_value = ts_data[i] + (i*4) # Obtient le timestamp associe
                    EEGraw_queue.append(array_data_list[i][:13])    


                if len(EEGraw_queue) > minimum_EEG_size:
                    
                    # Take the last minimum_EEG_size AKA window_size EEG lines (13 value per line)
                    last_epoch_raw = EEGraw_queue[-minimum_EEG_size:]
                   
                    # Preprocess raw data
                    X = np.array(last_epoch_raw).transpose(1,0)   # X shape = (Channels, Time) = (13, window_size)
                    X = filter_rawEEG(X, 0.5, 35)  
                    X = np.expand_dims(X, 0)    # X shape = (N, Channels, Time) = (1, 13, window_size)


                    # Returns predicted direction using a modular strategy 
                    direction = strategy.predict(X)

                    
                    print(f"Final Prediction = {direction}")

                    # POST request to server
                    if using_real_data:
                        requests.post(f"{server_ip}/push_direction", json={"direction": direction, "confidence": 1.0})


                    # Clean EEG queue once it's bigger than 1000
                    if len(EEGraw_queue) > 1000:
                        EEGraw_queue[:] = last_epoch_raw

            
            time.sleep(0.002)  # Ajouter un delai pour eviter d'occuper 100 du CPU

    except KeyboardInterrupt:
        # Arreter le thread proprement lors d'une interruption (Ctrl + C)
        if using_real_data:
            serial_flux.terminate() # Fermeture de la connexion EEG
            thread_a.join()
