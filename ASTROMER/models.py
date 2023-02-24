import tensorflow as tf
import numpy as np
import json
import os

from ASTROMER.core.astromer import get_ASTROMER, train, valid_step
from ASTROMER.utils import download_weights
from ASTROMER.core.data import load_numpy



class SingleBandEncoder(object):
    """
        This class is a transformer-based model that process the input and generate a fixed-size representation
        Since each light curve has two characteristics (magnitude and time) we transform into
        embeddings Z = 200x256.

        The maximum number of observations remain fixed and masked, so every Z had the same length even if some
        light curves are shorter than others.
    
        :param num_layer: Number of self-attention blocks or transformer layers in the encoder.
        :type num_layer: Integer

        :param d_model: Determines the dimensionality of the model's internal representation (must be divisible by 'num_heads').
        :type d_model: Integer

        :param num_heads: Number of attention heads used in an attention layer.
        :type num_heads: Integer

        :param dff: Number of neurons for the fully-connected layer applied after the attention layers. It consists of two linear transformations with a non-linear activation function in between.
        :type dff: Integer

        :param base: Value that defines the maximum and minimum wavelengths of the positional encoder (see equation 4 on Oliva-Donoso et al. 2022). Is used to define the range of positions the attention mechanism uses to compute the attention weights.
        :type base: Float32

        :param dropout: Regularization applied to output of the fully-connected layer to prevent overfitting. Randomly dropping out (i.e., setting to zero) some fraction of the input units in a layer during training.
        :type dropout: Float32

        :param maxlen: Maximum length to process in the encoder. It is used in the SingleBandEncoder class to limit the input sequences' length when passed to the transformer-based model.
        :type maxlen: Integer

        :param batch_size: Number of samples to be used in a forward pass. Note an epoch is completed when all batches were processed (default none).
        :type batch_size: Integer  
    """


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
        """
        The ‘fit()’ method trains ASTROMER for a given number of epochs. After each epoch, the model's performance is evaluated on the validation data, and the training stops if there is no improvement in a specified number of epochs (patience).

        :param train_batches: Training data already formatted as TF.data.Dataset
        :type train_batches: Object

        :param valid_batches: Validation data already formatted as TF.data.Dataset
        :type valid_batches: Object

        :param epochs: Number of training loops in where all light curves have been processed.
        :type epochs: Integer

        :param patience: The number of epochs with no improvement after which training will be stopped.
        :type patience: Integer

        :param lr: A float specifying the learning rate
        :type lr: Float32

        :param project_path: Path for saving weights and training logs

        :param verbose: if non zero, progress messages are printed. Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported."
        :type verbose: Integer

        :return:

        """

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
        """
        This method encodes a dataset of light curves into a fixed-dimensional embedding using the ASTROMER encoder.
        The method first checks the format of the dataset containing the light curves.

        Then, it loads the dataset using predefined functions from the ‘data’ module. In this part, if a light curve contains more than 200 observations, ASTROMER will divide it into shorter windows of 200 length.

        After loading data, the data pass through the encoder layer to obtain the embeddings.


        :param dataset: The input data to be encoded. It can be a list of numpy arrays or a tensorflow dataset.
        :type dataset:

        :param oids_list: list of object IDs. Since ASTROMER can only process fixed sequence of 200 observations, providing the IDs allows the model to concatenate windows when the length of the objects is larger than 200.
        :type oids_list: List

        :param labels: an optional list of labels for the objects associated to the input dataset.
        :type labels:

        :param batch_size: the number of samples to be used in a forward-pass within the encoder. Default is 1.
        :type batch_size:

        :param concatenate: a boolean indicating whether to concatenate the embeddings of objects with the same ID into a single vector.
        :type concatenate: Boolean
        :return:
        """

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
        """
        The ‘load_weights()’ method loads pre-trained parameters into the model architecture. The method loads the weights from the file located at {weights_folder}/weights directory, which is assumed to be in TensorFlow checkpoint format.

        :param weights_folder: the path to the folder containing the pre-trained weights.
        :return:
        """

        weights_path = '{}/weights'.format(weights_folder)
        self.model.load_weights(weights_path)

    def from_pretraining(cls, name='macho'):
        """
        Loads a pre-trained model with pre-trained weights for a specific astronomical dataset. This method allows users to easily load pre-trained models for astronomical time-series datasets and use them for their purposes.

        This method checks if you have the weights locally, if not then downloads and then uploads them.

        :param name: Corresponds to the name of the survey used to pre-train ASTROMER. The name of the survey should match with the name of the zip file in https://github.com/astromer-science/weights
        :return:
        """

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
