from Serial_class import serial_class
import threading
import time
import sys

if __name__ == '__main__':

    #Gestion port com
    serial_flux = serial_class()
    serial_flux.init_port()

    thread_a = threading.Thread(target=serial_flux.reception, name='ta')
    thread_a.start()

    time.sleep(2)
    cptEEG =0

    if not serial_flux.data:
        print('No serial flux , program is closing')
        serial_flux.terminate()
        sys.exit()



    try:
        while True:
            # Recuperer les donnees de la queue dans le main
            while not serial_flux.data_queue.empty():
                array_data_list , ts_data = serial_flux.data_queue.get()
                for i in range(len(array_data_list)):
                    ts_value = ts_data[i] + (i*4) # Obtient le timestamp associe
                    array_data = array_data_list[i]  # Obtient la ligne de donnees associee

                    print(ts_value, end=" ") 
                    for value in array_data:
                        print(round(value, 2), end=" ")   # Affiche chaque valeur sur la meme ligne avec un espace
                    print()
                    cptEEG +=1

            time.sleep(0.002)  # Ajouter un delai pour eviter d'occuper 100 du CPU
            if (cptEEG >250 *30):
                serial_flux.terminate() # Fermeture de la connexion EEG
                thread_a.join()
                break

    except KeyboardInterrupt:
        # Arreter le thread proprement lors d'une interruption (Ctrl + C)
        serial_flux.terminate() # Fermeture de la connexion EEG
        thread_a.join()



