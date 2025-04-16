"""
networks.py

This module contains the definition of Keras CNN models used with both CPU/GPU and Akida. 

Notes:
    - Model definition could be improved by parameterization to easily try different model configurations
    - Removing the separate functional definition of the Akida model
"""



from tensorflow import keras 
import tensorflow as tf


class CPU_NetCNN2D_CSP(keras.Model):
    """
    A simple 2D CNN used after the CSP transformation. This version is designed for CPU/GPU not for Akida. 
    Since Global Average Pooling is not compatible with Akida, but it seems useful for generalization.
    
    Parameters
    ----------
    n_classes : int
        Number of output classes for classification
    """


    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)

        self.n_classes = n_classes


        self.cnn1 = keras.Sequential(
            [
                keras.layers.Conv2D(filters=8, kernel_size=(3,11), strides=(1,7), padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(max_value=1000.0),

            ], name='cnn1')

        self.cnn2 = keras.Sequential(
            [
                keras.layers.Conv2D(filters=16, kernel_size=(3,7), strides=(1,5), padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(max_value=1000.0)
            ], name='cnn2')

        
        self.gap = keras.Sequential(
            [
                keras.layers.GlobalAveragePooling2D(),
            ]
        )

        self.fc = keras.Sequential( 
            [ 
                keras.layers.Flatten(),
                keras.layers.Dense(128),
                keras.layers.ReLU(max_value=1000.0),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(n_classes)

            ], name = 'fc')
        

    def call(self, x):
        temp =self.cnn1(x)
        temp = self.cnn2(temp)
        temp = self.gap(temp)
        temp = self.fc(temp)
        
        return temp
    

    # Re-define get_config and from_config for the serialized saving of the models.
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_classes': self.n_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    






class Akida_HWOnly_NetCNN2D_CSP(keras.Model):
    """
    A simple 2D CNN used after the CSP transformation.

    This version is designed for Akida Hardware Only mode (hw_only = True)
    (See list of hw_only mode constraints in README) 
    
    
    Parameters
    ----------
    n_classes : int
        Number of output classes for classification
    """


    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)

        self.n_classes = n_classes

        self.cnn1 = keras.Sequential(
            [
                keras.layers.Conv2D(filters=8, kernel_size=(7,7), strides=(3,3), padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(max_value=1000.0),

            ], name='cnn1')

        self.cnn2 = keras.Sequential(
            [
                keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(max_value=1000.0)
            ], name='cnn2')
        
        self.cnn3 = keras.Sequential(
            [
                keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(max_value=1000.0)
            ], name='cnn2')

    
        self.fc = keras.Sequential( 
            [ 
                keras.layers.Flatten(),
                keras.layers.Dense(128),
                keras.layers.ReLU(max_value=1000.0),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(n_classes)

            ], name = 'fc')
        


    def call(self, x):
        temp =self.cnn1(x)
        temp = self.cnn2(temp)
        temp = self.cnn3(temp)
        temp = self.fc(temp)
        
        return temp
    
    # Re-define get_config and from_config for the serialized saving of the models.
    # Might be not needed here, as the akida package will be in charge of saving these models
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_classes': self.n_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    



def build_functional_cnn2D(n_classes, input_shape = (7, 250, 1)):
    """
    convert the model to a functional form for Quantization/akida conversion.
    Needs to be changed according to changes in the original model above.

    Note: the model could be defined by default in this form.

    Parameters
    ----------
    n_classes : int
        Number of output classes for classification
    input_shape: tuple
        The input shape of the model: (Spatial_dimension, Time_dimension, channels), 
        defaults to (7, 250, 1): 
            7 is 3 CSP channels + zero padding for kernel_size=7
            250 is 1 second slice (sampling rate = 250)
            1 is the number of channels
    
    Returns
    -------
    functional_model
    
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # cnn1
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(7, 7), strides=(3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=1000.0)(x)

    # cnn2
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=1000.0)(x)

    # cnn3
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=1000.0)(x)
    
    # FC
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.ReLU(max_value=1000.0)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(n_classes)(x)
    
    functional_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return functional_model