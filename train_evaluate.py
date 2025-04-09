import dataset_NewEEG
import sys, os
from mne.decoding import CSP
import numpy as np

from tensorflow import keras
import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras.callbacks import ModelCheckpoint



import matplotlib.pyplot as plt

def train_evalute(config, X_train, Y_train, X_valid, Y_valid, model, csp=None, plot_graphs=True, train=True):
    
    

    #print("Training indices: ", train_index)
    #print("Validation indices: ", valid_index)


    # A better alternative is to slice the dataset before for preocessing time efficiency, however since our dataset is extremely small it doesn't matter in this case.
    X_train, Y_train = dataset_NewEEG.slice_EEG_epoch(config, X_train, Y_train)
    X_valid, Y_valid = dataset_NewEEG.slice_EEG_epoch(config, X_valid, Y_valid)


    #mu = X_train.mean()
    #sig = X_train.std()
    #X_train = (X_train - mu)/sig
    #X_valid = (X_valid - mu)/sig


    # This is used to not show CSP and other unneeded logs 
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    if csp == None:
        csp = CSP(n_components=config.n_csp_components, reg=None, log=None, norm_trace=False, transform_into='csp_space')
        X_train = csp.fit_transform(X_train, Y_train)

    else:
        X_train = csp.transform(X_train)


    X_valid = csp.transform(X_valid)

    sys.stdout.close()
    sys.stdout = old_stdout


    # Reshape input depending on model, 2D or 1D convs
    if config.model_type == '2D':
        X_train = np.expand_dims(X_train, 3)
        X_valid = np.expand_dims(X_valid, 3)

    else:
        X_train = X_train.transpose(0, 2, 1)
        X_valid = X_valid.transpose(0, 2, 1)


    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = len(config.used_classes))
    Y_valid = tf.keras.utils.to_categorical(Y_valid, num_classes = len(config.used_classes))


    model.compile(
            loss= CategoricalCrossentropy(from_logits=True, label_smoothing=config.label_smoothing),
            optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate, weight_decay = config.weight_decay), 
            metrics=[keras.metrics.CategoricalAccuracy(name='acc')]
    ) 


    if train:
        checkpoint_cb = ModelCheckpoint(os.path.join('saved_models', f'best_model_fold_fold.tf'), save_best_only=True, monitor='acc', mode='max', verbose=0, save_format='tf')
        history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=config.epochs, batch_size=config.batch_size, 
                            verbose=0, callbacks=[checkpoint_cb])
    else:
        history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=config.epochs, batch_size=config.batch_size, 
                            verbose=0)

    
    best_val_acc = max(history.history['acc'])



    if plot_graphs:
        plt.figure(figsize=(15,5))
        plt.title(f"| N_examples = {X_train.shape[0]} / {X_valid.shape[0]} | Best_val_acc={100*max(history.history['val_acc']):.2f}% ")

        plt.subplot(1,2,1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(["Train Loss", "Val Loss"])

        plt.subplot(1,2,2)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.legend(["Train Acc", "Val Acc"])

        plt.show(block=False)
        plt.pause(0.001)

    
    #model = tf.keras.models.load_model(os.path.join('saved_models', f'best_model_fold_fold.tf'))

    return model, csp, best_val_acc