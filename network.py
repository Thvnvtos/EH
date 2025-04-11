from tensorflow import keras 
import tensorflow as tf


class NetCNN2D_CSP_GAP(keras.Model):
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
    

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_classes': self.n_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    




# Kernel size should be 3, 5, 7 and symmetric
# strides should be 1, 2 or 3

class NetCNN2D_CSP(keras.Model):
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
                keras.layers.Conv2D(filters=16, kernel_size=(7,7), strides=(3,3), padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(max_value=1000.0)
            ], name='cnn2')
        

        self.cnn3 = keras.Sequential(
            [
                keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(3,3), padding='same'),
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
    

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_classes': self.n_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    



def build_functional_cnn2D(n_classes, input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # cnn1
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(7, 7), strides=(3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=1000.0)(x)

    # cnn2
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(7, 7), strides=(3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=1000.0)(x)

    # cnn3
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(3, 3), padding='same')(x)
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

















class NetCNN1D(keras.Model):
    '''
        Simple conv model for testing purposes
    '''
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)



        self.cnn1 = keras.Sequential(
            [
                keras.layers.Conv1D(filters=12, kernel_size=11, strides=7, groups=3),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(max_value=1000.0),
            ], name='cnn1')

        self.cnn2 = keras.Sequential(
            [

                keras.layers.Conv1D(filters=18, kernel_size=7, strides=5, groups=3),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(max_value=1000.0),
            ], name='cnn2')

        self.gap = keras.Sequential(
            [
                keras.layers.GlobalAveragePooling1D(), 
            ]
        )

        self.fc = keras.Sequential(
            [
                keras.layers.Flatten(),
                keras.layers.Dense(128),
                keras.layers.ReLU(max_value=1000.0),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(n_classes)

            ]
        , name = 'fc')



    def call(self, x):
        temp = self.cnn1(x)
        temp = self.cnn2(temp)
        temp = self.gap(temp)
        temp = self.fc(temp)
        return temp