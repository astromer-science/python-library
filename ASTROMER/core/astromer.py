import tensorflow as tf
import logging
import json
import os, sys
import pandas as pd
import numpy as np

from tensorflow.keras.layers  import Input, Dense
from tensorflow.keras         import Model

from .data                import load_numpy, inference_pipeline
from .components.decoder  import RegLayer, NSP_Regressor
from .utils               import download_weights
from .training.losses     import custom_rmse
from .training.metrics    import custom_r2
from .components.encoder  import Encoder

from tensorflow.keras.losses  import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
os.system('clear')


class BaseASTROMER(Model):
    def __init__(self, **kwargs):
        super(BaseASTROMER, self).__init__(**kwargs)

    def build(self, batch_size=None, max_obs=200, inp_dim=1):
        super(BaseASTROMER, self).build({'input': [batch_size, max_obs, 1],
                                     'mask_in': [batch_size, max_obs, 1],
                                     'times': [batch_size, max_obs, 1]})

    def call(self, inputs, training=False):
        raise NotImplementedError("Please Implement this method")

    def train_step(self, data):
        raise NotImplementedError("Please Implement this method")

    def test_step(self, data):
        raise NotImplementedError("Please Implement this method")

    def predict_step(self, data):
        raise NotImplementedError("Please Implement this method")

    def load_weights(self, filepath, **kwargs):
        self.load_json_config(filepath)
        super(BaseASTROMER, self).load_weights(os.path.join(filepath, 'weights.h5'),
                                                **kwargs)

    def load_json_config(self, filepath):
        conf_file = os.path.join(filepath, 'conf.json')
        with open(conf_file, 'r') as handle:
            conf = json.load(handle)

        self.encoder = Encoder(conf['layers'],
                               conf['head_dim'],
                               conf['heads'],
                               conf['dff'],
                               base=conf['base'],
                               rate=conf['dropout'],
                               name='encoder')
        self.regressor = RegLayer(name='regression')
        self.build(max_obs=conf['max_obs'])

    def encode(self, dataset, oids_list=None, labels=None, batch_size=50,
               concatenate=False):

        if isinstance(dataset, list):
            print('[INFO] Loading numpy arrays')
            dataset = load_numpy(dataset, ids=oids_list, labels=labels)
            dataset = inference_pipeline(dataset, batch_size=batch_size,
                                         max_obs=self.maxlen, drop_remainder=True,
                                         get_ids=True)

        att = self.predict(dataset)

        if concatenate:
            oids = tf.concat([oid for _, (_, oid) in dataset], axis=0)
            oids = np.array([str(o.numpy().decode('utf8') )for o in oids])
            unique_id = np.unique(oids)

            concat_att = []
            for id in unique_id:
                indices = np.where(oids == id)
                foo = np.concatenate(att[indices], 0)
                concat_att.append(foo)
            return concat_att
        return att

    def download_weights(self, remote, local):
        if not os.path.isdir(local):
            download_weights(remote, local)
        else:
            print('[INFO] Weights already downloaded')

    def from_pretrained(self):
        raise NotImplementedError("Please Implement this method")

class ASTROMER(BaseASTROMER):
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 num_heads=4,
                 dff=128,
                 base=10000,
                 dropout=0.1,
                 use_leak=False,
                 maxlen=200):

        super(ASTROMER, self).__init__()
        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               base=base,
                               rate=dropout,
                               use_leak=use_leak,
                               name='encoder')

        self.regressor = RegLayer(name='regression')
        self.maxlen = maxlen
        self.build(max_obs=maxlen)

    def compile(self, loss_rec=None, metric_rec=None, **kwargs):
        super(ASTROMER, self).compile(**kwargs)
        self.loss_rec = custom_rmse
        self.metric_rec = custom_r2

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training)
        x = self.regressor(x)
        return x

    def train_step(self, data):
        x, (y, _, mask) = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.loss_rec(y, y_pred, mask=mask)
            r2 = self.metric_rec(y, y_pred, mask=mask)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {'loss': loss, 'r2':r2}

    def test_step(self, data):
        x, (y, _, mask) = data
        y_pred = self(x, training=False)
        loss   = self.loss_rec(y, y_pred, mask=mask)
        r2     = self.metric_rec(y, y_pred, mask=mask)
        return {'loss': loss, 'r2':r2}

    def predict_step(self, data):
        x, _ = data
        return self.encoder(x)

    def from_pretrained(self, name):
        url = 'https://github.com/astromer-science/weights/raw/main/{}.zip'.format(name)
        target = './weights'
        self.download_weights(url, os.path.join(target, name))
        self.load_weights(os.path.join(target, name))
