import tensorflow as tf
import numpy as np
import json
import os

from .core.astromer import get_ASTROMER, train, valid_step
from .core.data import load_numpy
from .utils import download_weights

class SingleBandEncoder(object):
    """docstring for ASTROMER."""

    def __init__(self, num_layers=2,
                       d_model=200,
                       num_heads=2,
                       dff=256,
                       base=10000,
                       dropout=0.1,
                       maxlen=100,
                       batch_size=None):
        super(SingleBandEncoder, self).__init__()

        self.num_layers=num_layers
        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff
        self.base=base
        self.dropout=dropout
        self.maxlen=maxlen
        self.batch_size=batch_size

        self.model = get_ASTROMER(num_layers=self.num_layers,
                                  d_model=self.d_model,
                                  num_heads=self.num_heads,
                                  dff=self.dff,
                                  base=self.base,
                                  dropout=self.dropout,
                                  use_leak=False,
                                  no_train=False,
                                  maxlen=self.maxlen,
                                  batch_size=self.batch_size)
    def fit(self,
            train_batches,
            valid_batches,
            epochs=2,
            patience=40,
            lr = 1e-3,
            project_path='.',
            verbose=0):

        train(self.model,
              train_batches, valid_batches,
              patience=patience,
              exp_path=project_path,
              epochs=epochs,
              verbose=0,
              lr=lr)

    def encode(self,
               dataset,
               oids_list=None,
               labels=None,
               batch_size=1,
               concatenate=True):

        if isinstance(dataset, list):
            print('[INFO] Loading numpy arrays')
            dataset = load_numpy(dataset, ids=oids_list, labels=labels,
                                 batch_size=batch_size, sampling=False, shuffle=False,
                                 max_obs=self.maxlen)

        encoder = self.model.get_layer('encoder')

        att, lens, ids = [],[],[]
        for batch in dataset:
            emb = encoder(batch)

            sizes = tf.reduce_sum(1-batch['mask_in'], 1)
            sizes = tf.cast(sizes, tf.int32)

            att.append(emb)
            lens.append(sizes)
            ids.append([str(b.numpy().decode('utf8')) for b in batch['id']])

        if concatenate:
            att = np.concatenate(att, 0)
            lens = np.concatenate(lens, 0)
            ids = np.concatenate(ids, 0)

            final_att = []
            for oid in np.unique(ids):
                indices = np.where(ids == oid)
                foo = np.concatenate(att[indices], 0)
                goo = np.sum(lens[indices])
                final_att.append(foo[:goo])
            return final_att
        return att

    def load_weights(self, weights_folder):
        weights_path = '{}/weights'.format(weights_folder)
        self.model.load_weights(weights_path)

    def from_pretraining(cls, name='macho'):
        remote = 'https://github.com/astromer-science/weights/raw/main/{}.zip'.format(name)
        local = os.path.join('weights', name)

        if not os.path.isdir(local):
            download_weights(remote, local)
        else:
            print('[INFO] Weights already downloaded')

        conf_file = os.path.join(local, 'conf.json')
        with open(conf_file, 'r') as handle:
            conf = json.load(handle)

        model =  SingleBandEncoder(num_layers=conf['layers'],
                                   d_model=conf['head_dim'],
                                   num_heads=conf['heads'],
                                   dff=conf['dff'],
                                   base=conf['base'],
                                   dropout=conf['dropout'],
                                   maxlen=conf['max_obs'])
        model.load_weights(local)
        return model
