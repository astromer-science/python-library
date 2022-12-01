# test_load_weights.py
import numpy as np
import pytest
import sys
import os
import shutil

sys.path.append('../')
from ASTROMER.models import SingleBandEncoder
from ASTROMER.preprocessing import load_numpy

@pytest.fixture
def astromer():
    return SingleBandEncoder()

def test_load_weights(astromer):
    astromer = astromer.from_pretraining('macho')
    assert astromer.maxlen == 200

def test_encode(astromer):
    astromer = astromer.from_pretraining('macho')
    samples_collection = [np.ones([300, 3]), np.zeros([100, 3])]

    attention_vectors = astromer.encode(samples_collection,
                                        oids_list=['1', '2'],
                                        batch_size=2,
                                        concatenate=True)
    assert attention_vectors[1].shape == (100, 256)

def test_train(astromer):
    astromer = astromer.from_pretraining('macho')
    samples_collection = [np.ones([300, 3]), np.zeros([100, 3])]

    train_batches = load_numpy(samples_collection,
                               batch_size=2,
                               msk_frac=.5,
                               rnd_frac=.5,
                               same_frac=.5,
                               sampling=True,
                               shuffle=False,
                               max_obs=200)
    val_batches = load_numpy(samples_collection,
                             batch_size=2,
                             msk_frac=.5,
                             rnd_frac=.5,
                             same_frac=.5,
                             sampling=True,
                             shuffle=False,
                             max_obs=200)

    astromer.fit(train_batches,
                 val_batches,
                 project_path='./partial')

    assert os.path.isdir('./partial')

    shutil.rmtree('./partial')
