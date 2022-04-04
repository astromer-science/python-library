import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

from core.preprocess.masking import get_masked, set_random, pad_sequence
from core.preprocess.records import write_records, deserialize
from core.utils import standardize
from tqdm import tqdm
from time import time

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

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
    n_val = int(n_samples*val//2)

    if test_meta is not None:
        sub_test = test_meta
        sub_train = frame.iloc[:n_train]
        sub_val   = frame.iloc[n_train:]
    else:
        sub_train = frame.iloc[:n_train]
        sub_val   = frame.iloc[n_train:n_train+n_val]
        sub_test  = frame.iloc[n_train+n_val:]

    return ('train', sub_train), ('val', sub_val), ('test', test_meta)

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

    for cls_name, cls_meta in tqdm(cls_groups, total=len(cls_groups)):
        subsets = divide_training_subset(cls_meta,
                                         train=subsets_frac[0],
                                         val=subsets_frac[0],
                                         test_meta = test_subset)

        for subset_name, frame in subsets:
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

def sample_lc(input_dict, max_obs):
    '''
    Sample a random window of "max_obs" observations from the input sequence
    '''
    sequence = input_dict['input']
    serie_len = tf.shape(sequence)[0]
    pivot = 0

    def fn_true():
        pivot = tf.random.uniform([],
                                  minval=0,
                                  maxval=serie_len-max_obs+1,
                                  dtype=tf.int32)

        return tf.slice(sequence, [pivot,0], [max_obs, -1])


    def fn_false():
        return tf.slice(sequence, [0,0], [serie_len, -1])

    sequence = tf.cond(
                    tf.greater(serie_len, max_obs),
                    true_fn=lambda: fn_true(),
                    false_fn=lambda: fn_false()
                )

    sequence = standardize(sequence, return_mean=False)

    return sequence, input_dict['label'], input_dict['lcid']

def get_window(sequence, length, pivot, max_obs):
    pivot = tf.minimum(length-max_obs, pivot)
    pivot = tf.maximum(0, pivot)
    end = tf.minimum(length, max_obs)

    sliced = tf.slice(sequence, [pivot, 0], [end, -1])
    return sliced

def get_windows(input_dict, max_obs):
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

def randomize_lc(ds0, ds1, max_obs=200, proba=.5, inp_dim=3):
    '''
    cut sample and concat a random segment with some 'proba'bility
    '''
    seq_0, _, id_0 = ds0
    seq_1, _, _ = ds1

    # 0 = random / 1 = same
    label = tf.random.categorical([[proba, 1-proba]], 1)
    label = tf.squeeze(label)

    first  = tf.slice(seq_0, [0, 0],
                    [tf.minimum(max_obs//2, tf.shape(seq_0)[0]), -1])

    # tokens
    cls_tkn = tf.constant(-98., shape=[1, inp_dim])
    sep_tkn = tf.constant(-99., shape=[1, inp_dim])

    # getting mean delta time for the stitch-fix
    times = tf.slice(first, [0, 0], [-1, 1])
    delta = tf.reduce_mean(times[1:]-times[:-1])

    def fn_true():
        limit = tf.minimum(max_obs//2, tf.shape(seq_1)[0])

        second_time = tf.slice(seq_1, [0, 0], [limit, 1])
        second_rest = tf.slice(seq_1, [0, 1], [limit, -1])

        # stitch fix
        tau = times[-1] + delta
        second_time = second_time - second_time[0] + tau
        second = tf.concat([second_time, second_rest], 1)

        return tf.concat([cls_tkn, first, sep_tkn, second, sep_tkn], 0)


    def fn_false():
        return tf.concat([cls_tkn, seq_0, sep_tkn], 0)

    seq = tf.cond(
                tf.math.equal(label, 0),
                true_fn=lambda: fn_true(),
                false_fn=lambda: fn_false()
            )

    return seq, label, id_0

def mask_sample(x, y , i, msk_prob, rnd_prob, same_prob, max_obs):
    '''
    Pretraining formater
    '''

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
    shuffled_seq = tf.random.normal(tf.shape(seq_magn), 0, 0.5)

    seq_magn, mask_in = set_random(seq_magn,
                                   mask_in,
                                   shuffled_seq,
                                   rnd_prob,
                                   name='set_random')

    time_steps = tf.shape(seq_magn)[0]

    # Masked values should be 1, zero by default
    mask_in  = tf.reshape(mask_in, [time_steps, 1])
    mask_out = tf.reshape(mask_out, [time_steps, 1])
    # Using 1s in the "mask_in" tensor implies
    # not considering those values within the scaled dot product
    mask_in   = pad_sequence(mask_in, max_obs=max_obs, value=1.)
    # Using 1s in the "mask_out" tensor implies considering them
    # for loss calculation
    mask_out  = pad_sequence(mask_out, max_obs=max_obs, value=0.)
    # pad the rest
    orig_magn = pad_sequence(orig_magn, max_obs=max_obs, value=1.)
    seq_magn  = pad_sequence(seq_magn, max_obs=max_obs, value=1.)
    seq_time  = pad_sequence(seq_time, max_obs=max_obs, value=1.)

    input_dict = dict()
    input_dict['output']   = orig_magn
    input_dict['input']    = seq_magn
    input_dict['times']    = seq_time
    input_dict['mask_in']  = mask_in
    input_dict['mask_out'] = mask_out
    input_dict['length']   = time_steps
    input_dict['label']    = y
    input_dict['id']       = i

    return input_dict

def format_pt(input_dict, nsp=False):
    x = {
    'input':input_dict['input'],
    'times':input_dict['times'],
    'mask_in':input_dict['mask_in']
    }
    lab_one_hot = tf.one_hot(input_dict['label'], 2)
    if nsp:
        mask_out = tf.slice(input_dict['mask_out'], [1,0],[-1,-1])
        orig_magn = tf.slice(input_dict['output'], [1,0],[-1,-1])
    else:
        orig_magn = input_dict['output']
        mask_out = input_dict['mask_out']

    y = (orig_magn, lab_one_hot, mask_out)
    return x, y

def format_inference(input_dict, num_cls, get_ids=False):
    x = {
    'input':input_dict['input'],
    'times':input_dict['times'],
    'mask_in':input_dict['mask_in']
    }

    y = tf.one_hot(input_dict['label'], num_cls)
    if get_ids:
        y = (y, input_dict['id'])
    return x, y

def pretraining_pipeline_nsp(dataset_0, batch_size, max_obs=200, msk_frac=0.5,
                             rnd_frac=0.2, same_frac=0.2, nsp_proba=.5, inp_dim=3):
    '''
    Pretraining pipeline including NSP
    '''
    print('[INFO] Pretraining mode. Random {}-len windows'.format(max_obs))
    # Adjusting functionss
    fn_0 = adjust_fn(sample_lc, max_obs)
    fn_1 = adjust_fn(randomize_lc, max_obs, nsp_proba, inp_dim)
    max_obs = max_obs+3 # +TOKENS
    fn_2 = adjust_fn(mask_sample, msk_frac, rnd_frac, same_frac, max_obs)

    # Duplicate and shuffle the dataset to generate random segments
    dataset_1 = dataset_0.shuffle(10000)

    # get 200-len windows
    dataset_0 = dataset_0.map(fn_0)
    dataset_1 = dataset_1.map(fn_0)

    # zip datasets
    dataset = tf.data.Dataset.zip((dataset_0, dataset_1))
    dataset = dataset.map(fn_1)
    dataset = dataset.map(fn_2)
    dataset = dataset.map(format_pt)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def pretraining_pipeline(dataset, batch_size, max_obs=200, msk_frac=0.5, rnd_frac=0.2, same_frac=0.2, cache=False, take=-1):
    print('[INFO] Pretraining mode. Random {}-len windows'.format(max_obs))
    fn_0 = adjust_fn(sample_lc, max_obs)
    fn_1 = adjust_fn(mask_sample, msk_frac, rnd_frac, same_frac, max_obs)

    dataset = dataset.map(fn_0)
    dataset = dataset.map(fn_1)
    dataset = dataset.map(format_pt)
    dataset = dataset.batch(batch_size)

    if take != -1:
        print('[INFO] Taking {} batches'.format(take))
        dataset = dataset.take(take)

    if cache:
        print('[INFO] Cache activated')
        dataset = dataset.cache()

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def inference_pipeline(dataset, batch_size, max_obs=200, n_classes=1,
                       shuffle=False, drop_remainder=False, get_ids=False):
    print('[INFO] Inference mode. Cutting {}-len windows'.format(max_obs))
    fn_0 = adjust_fn(get_windows, max_obs)
    fn_1 = adjust_fn(mask_sample, 0., 0., 0., max_obs)
    fn_2 = adjust_fn(format_inference, n_classes, get_ids)

    dataset = dataset.map(fn_0)
    dataset = dataset.flat_map(lambda x,y,i: tf.data.Dataset.from_tensor_slices((x,y,i)))
    dataset = dataset.map(fn_1)
    dataset = dataset.map(fn_2)
    if shuffle:
        dataset = dataset.shuffle(100000)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def load_dataset(source, shuffle=False, repeat=1):
    rec_paths = []
    for folder in os.listdir(source):
        if folder.endswith('.csv'):
            continue
        for x in os.listdir(os.path.join(source, folder)):
            rec_paths.append(os.path.join(source, folder, x))

    dataset = tf.data.TFRecordDataset(rec_paths)
    dataset = dataset.map(deserialize)
    if shuffle:
        print('[INFO] Shuffling')
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(repeat)
    return dataset

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

def load_numpy(samples, ids=None, labels=None, shuffle=False, repeat=1):
    dataset = tf.data.Dataset.from_generator(lambda: create_generator(samples,labels,ids),
                                         output_types= {'input':tf.float32,
                                                        'label':tf.int32,
                                                        'lcid':tf.string,
                                                        'length':tf.int32},
                                         output_shapes={'input':(None,3),
                                                        'label':(),
                                                        'lcid':(),
                                                        'length':()})
    if shuffle:
        print('[INFO] Shuffling')
        dataset = dataset.shuffle(10000)

    dataset = dataset.repeat(repeat)
    return dataset
