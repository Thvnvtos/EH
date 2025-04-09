# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:42:19 2022

@author: lucas
"""
import queue as queue
import time
import datetime
import os
import numpy as np
import serial
import serial.tools.list_ports

map_freq = {

    182: 250,
    181: 500,
    180: 1000,
    179: 2000,
    178: 4000,
    177: 8000,
    176: 16000

}

map_gain = {

    40: 4,
    72: 8,
    82: 12,
    104: 24,
    5: 1,
    4: 40,
    8: 72,
    12: 82,
    24: 104,
    1: 5

}


class serial_class():
    def __init__(self):
        self._running = True
        self.serialPort = None

        self.bytes_data = []
        self.data = []
        self.timestamp = []
        self.time_adu = []
        self.vecteur_trigger = []
        self.adu = (4.5 * 1E6 / (24 * pow(2, 23)))

        self.nb_elec = 16
        self.gain = 1 # default gain
        self.Fe = 250
        self.name_subject = 'demo'
        self.bytes_capt1 = int('11111111', 2)  # blocage sensor 8 a 1
        self.bytes_capt2 = int('00011111', 2)  # blocage sensor 16 a 9 => ne pas prendre en compte les senseurs ]13 à 16]
        # self.bytes_capt2 = int('11111111', 2)  # blocage sensor 16 a 9

        self.data_queue = queue.Queue(maxsize=1000)  # Limiter la taille de la queue à 1000 messages soit 250*4s

    def terminate(self):
        self._running = False
        self.serialPort.write('stop\n'.encode('utf-8'))
        time.sleep(0.1)
        self.serialPort.close()

    def init_port(self):
        ports = serial.tools.list_ports.comports()
        com = None
        for port in ports:
            print(f"Port: {port.device}")
            print(f"  Description : {port.description}")
            print(f"  Hardware ID : {port.hwid}")
            print(f"  Vendor ID   : {port.vid}")
            print(f"  Product ID  : {port.pid}")
            print(f"  Manufacturer: {port.manufacturer}")
            print(f"  Serial Num  : {port.serial_number}")
            print(f"  Location    : {port.location}")
            print(f"  Product     : {port.product}")
            print()  # Ligne vide entre chaque port

            if "CP210x" in port.description: # si plusieur port com prendre celui qui se nomme CP210x
                com = port.device  # 'COM'
        print("port :", com)
        self.serialPort = serial.Serial(
            port = com, \
            baudrate = 2000000, \
            bytesize = serial.EIGHTBITS , timeout = 1)

    def change_gain(self, new_gain):
        self.gain = new_gain
        self.adu = (4.5 * 1E6 / (new_gain * pow(2, 23))) # Calcul conversion ADU en tension

    def reception(self):
        self.serialPort.write('connect\n'.encode('utf-8')) #Commande de connexion
        time.sleep(0.5)
        self.serialPort.write(("{\"cmd\":\"cut\",\"stateS1_8\":" + str(self.bytes_capt1) + ",\"stateS9_16\":" + str(
            self.bytes_capt2) + "}\n").encode('utf-8'))
        time.sleep(0.3)
        self.serialPort.write(("{\"cmd\":\"gain\", \"level\":" + str(map_gain[self.gain]) + "}\n").encode('utf-8'))
        time.sleep(0.3)
        self.serialPort.flush()
        while (self._running):
            # Wait until there is data waiting in the serial buffer
            if (self.serialPort.in_waiting > 0):
                # Read data out of the buffer until a carraige return / new line is found
                serialString = self.serialPort.read_until(b'\n')
                # print(serialString)
                self.bytes_data.append(serialString)
                if serialString[2:4] == b'nb':
                    self.read_header(serialString)
                if serialString[2:4] == b'ts':
                    t, d = self.read_line(serialString)
                    ligneData =np.array_split(self.adu_to_data(d, self.nb_elec), 4)# extract ligne data
                    self.data.extend(ligneData)
                    self.time_adu.extend([t] * 4)
                    self.vecteur_trigger.extend([0] * 4)
                    self.timestamp.extend([time.time()] * 4)
                    tpsEEG0 = [t]*4
                    tpsEEG1 = tpsEEG0
                    tpsEEG2 = tpsEEG0
                    tpsEEG3 = tpsEEG0

                    tps = [tpsEEG0 , tpsEEG1 , tpsEEG2 , tpsEEG3]

                    #Envoi des data EEG pour le Main
                    try:
                        self.data_queue.put((ligneData,tpsEEG0), timeout=1)  # Mettre les donnees dans la queue avec un timeout
                    except queue.Full:
                        print("Queue is full, dropping data")

    #Lecture entete de la config du casque
    def read_header(self, tram):
        
        str_tram = tram.decode()
        list_tram = str_tram.split('"')
        for i in range(len(list_tram)):
            if list_tram[i] == 'nb':
                nb = list_tram[i + 1]
                nb = nb.replace(':', '')
                nb = nb.replace(',', '')
                self.nb_elec = int(nb)

            if list_tram[i] == 'fq':
                fq = list_tram[i + 1]
                fq = fq.replace(':', '')
                fq = fq.replace(',', '')
                fq = int(fq)
                self.Fe = map_freq[fq]
            if list_tram[i] == 'gn':
                gn = list_tram[i + 1]
                gn = gn.replace(':', '')
                gn = gn.replace(',', '')
                gn = int(gn)
                self.change_gain(map_gain[gn])


    #Lecture de la trame des data EEG
    def read_line(self, tram):
        str_tram = tram.decode()
        list_tram = str_tram.split('"')
        timestamp = 0
        data = []
        try:
            for i in range(len(list_tram)):
                if list_tram[i] == 'ts':
                    timestamp = list_tram[i + 1]
                    timestamp = timestamp.replace(':', '')
                    timestamp = timestamp.replace(',', '')
                    timestamp = int(timestamp)
                if list_tram[i] == 'sx':
                    data_str = list_tram[i + 1]
                    data_str = data_str.replace(':', '')
                    data_str = data_str.replace('[', '')
                    data_str = data_str.replace(']', '')
                    data = data_str.split(',')
                    data.pop(-1)
                    data = [int(numeric_string) for numeric_string in data]



        except:
            timestamp = self.timestamp[-1]
            data = np.concatenate(self.data[-4:])

        return timestamp, data

    #Conversion data ADU
    def adu2uV(self, data_adu):
        # print(data_adu)
        data_uV =  data_adu * self.adu
        return data_uV

    #Extraction des data EEG de la trame
    def adu_to_data(self, data, nb_elec):
        list_adu = []
        for i in range(4):
            for j in range(nb_elec):
                temp_adu = data[j]
                if i > 0:
                    temp_adu = temp_adu - data[j + (i * nb_elec)]

                temp_adu =  self.adu2uV(temp_adu)
                list_adu.append(temp_adu)

        return list_adu






