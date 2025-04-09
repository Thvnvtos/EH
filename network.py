from tensorflow import keras 



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
    


class NetCNN2D_CSP(keras.Model):
    '''
        Simple conv model for testing purposes
    '''
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
                keras.layers.ReLU(max_value=1000.0),

                #keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same'),
                #keras.layers.BatchNormalization(),
                #keras.layers.ReLU(max_value=1000.0)
            ], name='cnn2')

        
        #elf.gap = keras.Sequential(
        #    [
        #        keras.layers.GlobalAveragePooling2D(),
        #    ]
        #)

        self.fc = keras.Sequential( 
            [ 
                keras.layers.Flatten(),
                keras.layers.Dense(128),
                keras.layers.ReLU(max_value=1000.0),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(n_classes)

            ], name = 'fc')
        

        #self.model = keras.Sequential([self.cnn1, self.cnn2, self.gap, self.fc])


    def call(self, x):
        temp =self.cnn1(x)
        temp = self.cnn2(temp)
        #temp = self.gap(temp)
        temp = self.fc(temp)
        
        #temp = self.model(x)
        
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