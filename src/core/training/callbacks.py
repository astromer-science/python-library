import tensorflow as tf
import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


def get_callbacks(project_dir, patience=40):
    ckp_callback = ModelCheckpoint(
                    filepath=os.path.join(project_dir, 'weights.h5'),
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True)
    esp_callback = EarlyStopping(monitor ='val_loss',
                                 mode = 'min',
                                 patience = patience,
                                 restore_best_weights=True)
    tsb_callback = TensorBoard(
                    log_dir = os.path.join(project_dir, 'logs'),
                    histogram_freq=1,
                    write_graph=True)


    return [ esp_callback, tsb_callback, ckp_callback]
