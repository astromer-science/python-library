import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

from core.masking import get_masked, set_random, get_padding_mask
from core.utils import standardize
from joblib import Parallel, delayed
from tqdm import tqdm

from time import time
from joblib import wrap_non_picklable_objects

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_example(lcid, label, lightcurve):
    """
    Create a record example from numpy values.

    Args:
        lcid (string): object id
        label (int): class code
        lightcurve (numpy array): time, magnitudes and observational error

    Returns:
        tensorflow record
    """

    f = dict()

    dict_features={
    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(lcid).encode()])),
    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[lightcurve.shape[0]])),
    }
    element_context = tf.train.Features(feature = dict_features)

    dict_sequence = {}
    for col in range(lightcurve.shape[1]):
        seqfeat = _float_feature(lightcurve[:, col])
        seqfeat = tf.train.FeatureList(feature = [seqfeat])
        dict_sequence['dim_{}'.format(col)] = seqfeat

    element_lists = tf.train.FeatureLists(feature_list=dict_sequence)
    ex = tf.train.SequenceExample(context = element_context,
                                  feature_lists= element_lists)
    return ex

def divide_training_subset(frame, train, val, test_meta):
    """
    Divide the dataset into train, validation and test subsets.
    Notice that:
        test = 1 - (train + val)
    Args:
        frame (Dataframe): Dataframe following the astro-standard format
        dest (string): Record destination.
        train (float): train fraction
        val (float): validation fraction
    Returns:
        tuple x3 : (name of subset, subframe with metadata)
    """

    frame = frame.sample(frac=1)
    n_samples = frame.shape[0]

    n_train = int(n_samples*train)
    n_val = int(n_samples*val)

    if test_meta is not None:
        sub_test = test_meta
        sub_train = frame.iloc[:n_train]
        sub_val   = frame.iloc[n_train:]
    else:
        sub_train = frame.iloc[:n_train]
        sub_val   = frame.iloc[n_train:n_train+n_val]
        sub_test  = frame.iloc[n_train+n_val:]

    return ('train', sub_train), ('val', sub_val), ('test', sub_test)

@wrap_non_picklable_objects
def process_lc2(row, source, unique_classes, **kwargs):
    path  = row['Path'].split('/')[-1]
    label = list(unique_classes).index(row['Class'])
    lc_path = os.path.join(source, path)

    observations = pd.read_csv(lc_path, **kwargs)
    observations.columns = ['mjd', 'mag', 'errmag']
    observations = observations.dropna()
    observations.sort_values('mjd')
    observations = observations.drop_duplicates(keep='last')

    numpy_lc = observations.values

    return row['ID'], label, numpy_lc

def process_lc3(lc_index, label, numpy_lc, writer):
    try:
        ex = get_example(lc_index, label, numpy_lc)
        writer.write(ex.SerializeToString())
    except:
        print('[INFO] {} could not be processed'.format(lc_index))

def write_records(frame, dest, max_lcs_per_record, source, unique, n_jobs=None, max_obs=200, **kwargs):
    # Get frames with fixed number of lightcurves
    collection = [frame.iloc[i:i+max_lcs_per_record] \
                  for i in range(0, frame.shape[0], max_lcs_per_record)]

    for counter, subframe in enumerate(collection):
        var = Parallel(n_jobs=n_jobs)(delayed(process_lc2)(row, source, unique, **kwargs) \
                                    for k, row in subframe.iterrows())

        with tf.io.TFRecordWriter(dest+'/chunk_{}.record'.format(counter)) as writer:
            for counter2, data_lc in enumerate(var):
                process_lc3(*data_lc, writer)

def create_dataset(meta_df,
                   source='data/raw_data/macho/MACHO/LCs',
                   target='data/records/macho/',
                   n_jobs=None,
                   subsets_frac=(0.5, 0.25),
                   test_subset=None,
                   max_lcs_per_record=100,
                   **kwargs): # kwargs contains additional arguments for the read_csv() function
    os.makedirs(target, exist_ok=True)

    bands = meta_df['Band'].unique()
    if len(bands) > 1:
        b = input('Filters {} were found. Type one to continue'.format(' and'.join(bands)))
        meta_df = meta_df[meta_df['Band'] == b]

    unique, counts = np.unique(meta_df['Class'], return_counts=True)
    info_df = pd.DataFrame()
    info_df['label'] = unique
    info_df['size'] = counts
    info_df.to_csv(os.path.join(target, 'objects.csv'), index=False)

    # Separate by class
    cls_groups = meta_df.groupby('Class')

    test_already_written = False
    if test_subset is not None:
        for cls_name, frame in test_subset.groupby('Class'):
            dest = os.path.join(target, 'test', cls_name)
            os.makedirs(dest, exist_ok=True)
            write_records(frame, dest, max_lcs_per_record, source, unique, n_jobs, **kwargs)
        test_already_written = True

    for cls_name, cls_meta in tqdm(cls_groups, total=len(cls_groups)):
        subsets = divide_training_subset(cls_meta,
                                         train=subsets_frac[0],
                                         val=subsets_frac[1],
                                         test_meta = test_subset)

        for subset_name, frame in subsets:
            if test_already_written and subset_name == 'test':continue
            dest = os.path.join(target, subset_name, cls_name)
            os.makedirs(dest, exist_ok=True)
            write_records(frame, dest, max_lcs_per_record, source, unique, n_jobs, **kwargs)
# ==============================
# ====== LOADING FUNCTIONS =====
# ==============================
def adjust_fn(func, *arguments):
    def wrap(*args, **kwargs):
        result = func(*args, *arguments)
        return result
    return wrap

def deserialize(sample):
    """
    Read a serialized sample and convert it to tensor
    Context and sequence features should match with the name used when writing.
    Args:
        sample (binary): serialized sample

    Returns:
        type: decoded sample
    """
    context_features = {'label': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'length': tf.io.FixedLenFeature([],dtype=tf.int64),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = dict()
    for i in range(3):
        sequence_features['dim_{}'.format(i)] = tf.io.VarLenFeature(dtype=tf.float32)

    context, sequence = tf.io.parse_single_sequence_example(
                            serialized=sample,
                            context_features=context_features,
                            sequence_features=sequence_features
                            )

    input_dict = dict()
    input_dict['lcid']   = tf.cast(context['id'], tf.string)
    input_dict['length'] = tf.cast(context['length'], tf.int32)
    input_dict['label']  = tf.cast(context['label'], tf.int32)

    casted_inp_parameters = []
    for i in range(3):
        seq_dim = sequence['dim_{}'.format(i)]
        seq_dim = tf.sparse.to_dense(seq_dim)
        seq_dim = tf.cast(seq_dim, tf.float32)
        casted_inp_parameters.append(seq_dim)


    sequence = tf.stack(casted_inp_parameters, axis=2)[0]
    input_dict['input'] = sequence
    return input_dict

def sample_lc(sample, max_obs, deserialize=True):
    '''
    Sample a random window of "max_obs" observations from the input sequence
    '''
    if deserialize:
        input_dict = deserialize(sample)
    else:
        input_dict = sample

    sequence = input_dict['input']

    serie_len = tf.shape(sequence)[0]

    pivot = 0
    if tf.greater(serie_len, max_obs):
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-max_obs+1,
                                  dtype=tf.int32)

        sequence = tf.slice(sequence, [pivot,0], [max_obs, -1])
    else:
        sequence = tf.slice(sequence, [0,0], [serie_len, -1])

    input_dict['sequence'] = sequence
    return sequence, input_dict['label'], input_dict['lcid']

def get_window(sequence, length, pivot, max_obs):
    pivot = tf.minimum(length-max_obs, pivot)
    pivot = tf.maximum(0, pivot)
    end = tf.minimum(length, max_obs)

    sliced = tf.slice(sequence, [pivot, 0], [end, -1])
    return sliced

def get_windows(sample, max_obs, deserialize=True):
    if deserialize:
        input_dict = deserialize(sample)
    else:
        input_dict = sample

    sequence = input_dict['input']
    rest = input_dict['length']%max_obs

    pivots = tf.tile([max_obs], [tf.cast(input_dict['length']/max_obs, tf.int32)])
    pivots = tf.concat([[0], pivots], 0)
    pivots = tf.math.cumsum(pivots)

    splits = tf.map_fn(lambda x: get_window(sequence,
                                            input_dict['length'],
                                            x,
                                            max_obs),  pivots,
                       infer_shape=False,
                       fn_output_signature=(tf.float32))

    # aqui falta retornar labels y oids
    y = tf.tile([input_dict['label']], [len(splits)])
    ids = tf.tile([input_dict['lcid']], [len(splits)])

    return splits, y, ids

def mask_sample(x, y , i, msk_prob, rnd_prob, same_prob, max_obs):
    '''
    Pretraining formater
    '''
    x = standardize(x, return_mean=False)

    seq_time = tf.slice(x, [0, 0], [-1, 1])
    seq_magn = tf.slice(x, [0, 1], [-1, 1])
    seq_errs = tf.slice(x, [0, 2], [-1, 1])

    # Save the true values
    orig_magn = seq_magn

    # [MASK] values
    mask_out = get_masked(seq_magn, msk_prob)

    # [MASK] -> Same values
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_out,
                                   seq_magn,
                                   same_prob,
                                   name='set_same')

    # [MASK] -> Random value
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_in,
                                   tf.random.shuffle(seq_magn),
                                   rnd_prob,
                                   name='set_random')

    time_steps = tf.shape(seq_magn)[0]

    mask_out = tf.reshape(mask_out, [time_steps, 1])
    mask_in = tf.reshape(mask_in, [time_steps, 1])

    if time_steps < max_obs:
        mask_fill = tf.ones([max_obs - time_steps, 1], dtype=tf.float32)
        mask_out  = tf.concat([mask_out, 1-mask_fill], 0)
        mask_in   = tf.concat([mask_in, mask_fill], 0)
        seq_magn   = tf.concat([seq_magn, 1-mask_fill], 0)
        seq_time   = tf.concat([seq_time, 1-mask_fill], 0)
        orig_magn   = tf.concat([orig_magn, 1-mask_fill], 0)

    input_dict = dict()
    input_dict['output']   = orig_magn
    input_dict['input']    = seq_magn
    input_dict['times']    = seq_time
    input_dict['mask_out'] = mask_out
    input_dict['mask_in']  = mask_in
    input_dict['length']   = time_steps
    input_dict['label']    = y
    input_dict['id']       = i

    return input_dict

def format_label(input_dict, num_cls):
    x = {
    'input':input_dict['input'],
    'times':input_dict['times'],
    'mask_in':input_dict['mask_in']
    }
    y = tf.one_hot(input_dict['label'], num_cls)
    return x, y

def pretraining_records(source, batch_size, max_obs=100, msk_frac=0.2,
                        rnd_frac=0.1, same_frac=0.1, sampling=False,
                        shuffle=False, n_classes=-1, repeat=1):
    """
    Pretraining data loader.
    This method build the ASTROMER input format.
    ASTROMER format is based on the BERT masking strategy.

    Args:
        source (string): Record folder
        batch_size (int): Batch size
        no_shuffle (bool): Do not shuffle training and validation dataset
        max_obs (int): Max. number of observation per serie
        msk_frac (float): fraction of values to be predicted ([MASK])
        rnd_frac (float): fraction of [MASKED] values to replace with random values
        same_frac (float): fraction of [MASKED] values to replace with true values

    Returns:
        Tensorflow Dataset: Iterator withg preprocessed batches
    """
    rec_paths = []
    for folder in os.listdir(source):
        if folder.endswith('.csv'):
            continue
        for x in os.listdir(os.path.join(source, folder)):
            rec_paths.append(os.path.join(source, folder, x))

    if sampling:
        fn_0 = adjust_fn(sample_lc, max_obs)
    else:
        fn_0 = adjust_fn(get_windows, max_obs)

    fn_1 = adjust_fn(mask_sample, msk_frac, rnd_frac, same_frac, max_obs)

    dataset = tf.data.TFRecordDataset(rec_paths)
    dataset = dataset.repeat(repeat)
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(fn_0)

    if not sampling:
        dataset = dataset.flat_map(lambda x,y,i: tf.data.Dataset.from_tensor_slices((x,y,i)))

    dataset = dataset.map(fn_1)

    if n_classes!=-1:
        print('[INFO] Processing labels')
        fn_2 = adjust_fn(format_label, n_classes)
        dataset = dataset.map(fn_2)

    dataset = dataset.padded_batch(batch_size).cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def adjust_fn_clf(func, max_obs):
    def wrap(*args, **kwargs):
        result = func(*args, max_obs)
        return result
    return wrap

def create_generator(list_of_arrays, labels=None, ids=None):

    if ids is None:
        ids = list(range(len(list_of_arrays)))
    if labels is None:
        labels = list(range(len(list_of_arrays)))

    for i, j, k in zip(list_of_arrays, labels, ids):
        yield {'input': i,
               'label':int(j),
               'lcid':str(k),
               'length':int(i.shape[0])}

def load_numpy(samples,
               ids=None,
               labels=None,
               batch_size=1,
               shuffle=False,
               sampling=False,
               max_obs=100,
               msk_frac=0.,
               rnd_frac=0.,
               same_frac=0.,
               repeat=1):
    if sampling:
        fn_0 = adjust_fn(sample_lc, max_obs, False)
    else:
        fn_0 = adjust_fn(get_windows, max_obs, False)

    fn_1 = adjust_fn(mask_sample, msk_frac, rnd_frac, same_frac, max_obs)

    dataset = tf.data.Dataset.from_generator(lambda: create_generator(samples,labels,ids),
                                         output_types= {'input':tf.float32,
                                                        'label':tf.int32,
                                                        'lcid':tf.string,
                                                        'length':tf.int32},
                                         output_shapes={'input':(None,3),
                                                        'label':(),
                                                        'lcid':(),
                                                        'length':()})
    dataset = dataset.repeat(repeat)
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(fn_0)
    if not sampling:
        dataset = dataset.flat_map(lambda x,y,i: tf.data.Dataset.from_tensor_slices((x,y,i)))
    dataset = dataset.map(fn_1)
    dataset = dataset.padded_batch(batch_size).cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
