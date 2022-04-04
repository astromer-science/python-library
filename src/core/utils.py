import tensorflow as tf
import seaborn as sns
import numpy as np
import json
import os

from tensorboard.backend.event_processing import event_accumulator
from tensorflow.python.lib.io import tf_record
from tensorflow.core.util import event_pb2
from datetime import datetime

def draw_graph(model, dataset, writer, logdir=''):
    '''Decorator that reports store fn graph.'''

    @tf.function
    def fn(x):
        x = model(x)

    tf.summary.trace_on(graph=True, profiler=False)
    fn(dataset)
    with writer.as_default():
        tf.summary.trace_export(
            name='model',
            step=0,
            profiler_outdir=logdir)


def save_scalar(writer, value, step, name=''):
    with writer.as_default():
        tf.summary.scalar(name, value.result(), step=step)

def plot_cm(cm, ax, title='CM', fontsize=15, cbar=False, yticklabels=True, class_names=None):
    '''
    Plot Confusion Matrix
    '''
    labels = np.zeros_like(cm, dtype=np.object)
    mask = np.ones_like(cm, dtype=np.bool)
    for (row, col), value in np.ndenumerate(cm):
        if value != 0.0:
            mask[row][col] = False
        if value < 0.01:
            labels[row][col] = '< 1%'
        else:
            labels[row][col] = '{:2.1f}%'.format(value*100)

    ax = sns.heatmap(cm, annot = labels, fmt = '',
                     annot_kws={"size": fontsize},
                     cbar=cbar,
                     ax=ax,
                     linecolor='white',
                     linewidths=1,
                     vmin=0, vmax=1,
                     cmap='Blues',
                     mask=mask,
                     yticklabels=yticklabels)

    try:
        if yticklabels and class_names is not None:
            ax.set_yticklabels(class_names, rotation=0, fontsize=fontsize+1)
            ax.set_xticklabels(class_names, rotation=90, fontsize=fontsize+1)
    except:
        pass
    ax.set_title(title, fontsize=fontsize+5)

    ax.axhline(y=0, color='k',linewidth=4)
    ax.axhline(y=cm.shape[1], color='k',linewidth=4)
    ax.axvline(x=0, color='k',linewidth=4)
    ax.axvline(x=cm.shape[0], color='k',linewidth=4)

    return ax

def get_folder_name(path, prefix=''):
    """
    Look at the current path and change the name of the experiment
    if it is repeated

    Args:
        path (string): folder path
        prefix (string): prefix to add

    Returns:
        string: unique path to save the experiment
"""

    if prefix == '':
        prefix = path.split('/')[-1]
        path = '/'.join(path.split('/')[:-1])

    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    if prefix not in folders:
        path = os.path.join(path, prefix)
    elif not os.path.isdir(os.path.join(path, '{}_0'.format(prefix))):
        path = os.path.join(path, '{}_0'.format(prefix))
    else:
        n = sorted([int(f.split('_')[-1]) for f in folders if '_' in f[-2:]])[-1]
        path = os.path.join(path, '{}_{}'.format(prefix, n+1))

    return path

def standardize(tensor, axis=0, return_mean=False):
    """
    Standardize a tensor subtracting the mean

    Args:
        tensor (1-dim tensorflow tensor): values
        axis (int): axis on which we calculate the mean
        return_mean (bool): output the mean of the tensor
                            turning on the original scale
    Returns:
        tensor (1-dim tensorflow tensor): standardize tensor
    """
    mean_value = tf.reduce_mean(tensor, axis, name='mean_value')
    z = tensor - tf.expand_dims(mean_value, axis)

    if return_mean:
        return z, mean_value
    else:
        return z

def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

def get_metrics(path_logs):
    train_logs = [x for x in os.listdir(path_logs) if x.endswith('.v2')][0]
    path_train = os.path.join(path_logs, train_logs)

    ea = event_accumulator.EventAccumulator(path_train)
    ea.Reload()

    metrics = {tag:[] for tag in ea.Tags()['tensors']}
    for e in my_summary_iterator(path_train):
        for v in e.summary.value:
            if v.tag == 'model': continue
            value = tf.make_ndarray(v.tensor)
            metrics[v.tag].append(value)
    return metrics

def dict_to_json(dictonary, project_path):
    os.makedirs(project_path, exist_ok=True)
    conf_file = os.path.join(project_path, 'conf.json')
    varsdic = vars(dictonary)
    now = datetime.now()
    varsdic['exp_date'] = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(conf_file, 'w') as json_file:
        json.dump(varsdic, json_file, indent=4)
