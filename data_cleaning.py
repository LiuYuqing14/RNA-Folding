import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil
import math
import pandas as pd
import gc
import os
import background

tffiles_path = 'stanford-ribonanza-rna-folding-data'
tffiles = [f'{tffiles_path}/{x}.tfrecord' for x in range(164)]

# Decoding tensorflow data
def decode_tfrec(record_bytes):
    schema = {}
    schema["id"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["seq"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["dataset_name_2A3"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["dataset_name_DMS"] = tf.io.VarLenFeature(dtype=tf.string)
    schema["reads_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reads_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["signal_to_noise_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["signal_to_noise_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["SN_filter_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["SN_filter_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_error_2A3"] = tf.io.VarLenFeature(dtype=tf.float32)
    schema["reactivity_error_DMS"] = tf.io.VarLenFeature(dtype=tf.float32)
    features = tf.io.parse_single_example(record_bytes, schema)

    sample_id = tf.sparse.to_dense(features["id"])
    seq = tf.sparse.to_dense(features["seq"])
    dataset_name_2A3 = tf.sparse.to_dense(features["dataset_name_2A3"])
    dataset_name_DMS = tf.sparse.to_dense(features["dataset_name_DMS"])
    reads_2A3 = tf.sparse.to_dense(features["reads_2A3"])
    reads_DMS = tf.sparse.to_dense(features["reads_DMS"])
    signal_to_noise_2A3 = tf.sparse.to_dense(features["signal_to_noise_2A3"])
    signal_to_noise_DMS = tf.sparse.to_dense(features["signal_to_noise_DMS"])
    SN_filter_2A3 = tf.sparse.to_dense(features["SN_filter_2A3"])
    SN_filter_DMS = tf.sparse.to_dense(features["SN_filter_DMS"])
    reactivity_2A3 = tf.sparse.to_dense(features["reactivity_2A3"])
    reactivity_DMS = tf.sparse.to_dense(features["reactivity_DMS"])
    reactivity_error_2A3 = tf.sparse.to_dense(features["reactivity_error_2A3"])
    reactivity_error_DMS = tf.sparse.to_dense(features["reactivity_error_DMS"])

    out = {}
    out['seq']  = seq
    out['SN_filter_2A3']  = SN_filter_2A3
    out['SN_filter_DMS']  = SN_filter_DMS
    out['reads_2A3']  = reads_2A3
    out['reads_DMS']  = reads_DMS
    out['signal_to_noise_2A3']  = signal_to_noise_2A3
    out['signal_to_noise_DMS']  = signal_to_noise_DMS
    out['reactivity_2A3']  = reactivity_2A3
    out['reactivity_DMS']  = reactivity_DMS
    return out

# Filtering
def f1(): return True
def f2(): return False

def filter_function_1(x):
    SN_filter_2A3 = x['SN_filter_2A3']
    SN_filter_DMS = x['SN_filter_DMS']
    return tf.cond((SN_filter_2A3 == 1) and (SN_filter_DMS == 1) , true_fn=f1, false_fn=f2)

def filter_function_2(x):
    reads_2A3 = x['reads_2A3']
    reads_DMS = x['reads_DMS']
    signal_to_noise_2A3 = x['signal_to_noise_2A3']
    signal_to_noise_DMS = x['signal_to_noise_DMS']
    cond = (reads_2A3>100 and signal_to_noise_2A3>0.75) or (reads_DMS>100 and signal_to_noise_DMS>0.75)
    return tf.cond(cond, true_fn=f1, false_fn=f2)

# filtering nan values
def nan_below_filter(x):
    reads_2A3 = x['reads_2A3']
    reads_DMS = x['reads_DMS']
    signal_to_noise_2A3 = x['signal_to_noise_2A3']
    signal_to_noise_DMS = x['signal_to_noise_DMS']
    reactivity_2A3 = x['reactivity_2A3']
    reactivity_DMS = x['reactivity_DMS']

    if reads_2A3<100 or signal_to_noise_2A3<0.75:
        reactivity_2A3 = np.nan+reactivity_2A3
    if reads_DMS<100 or signal_to_noise_DMS<0.75:
        reactivity_DMS = np.nan+reactivity_DMS

    x['reactivity_2A3'] = reactivity_2A3
    x['reactivity_DMS'] = reactivity_DMS
    return x

# If only Samples is below the threshold, it becomes a Nan array, ignoring in the loss function.
def concat_target(x):
    reactivity_2A3 = x['reactivity_2A3']
    reactivity_DMS = x['reactivity_DMS']
    target = tf.concat([reactivity_2A3[..., tf.newaxis], reactivity_DMS[..., tf.newaxis]], axis = 1)
    target = tf.clip_by_value(target, 0, 1)
    return x['seq'], target


def get_tfrec_dataset(tffiles, shuffle, batch_size, cache=False, to_filter=False,
                      calculate_sample_num=True, to_repeat=False):
    ds = tf.data.TFRecordDataset(
        tffiles, num_parallel_reads=tf.data.AUTOTUNE, compression_type='GZIP').prefetch(tf.data.AUTOTUNE)

   # Apply filter functions
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    if to_filter == 'filter_1':
        ds = ds.filter(filter_function_1)
    elif to_filter == 'filter_2':
        ds = ds.filter(filter_function_2)
    ds = ds.map(nan_below_filter, tf.data.AUTOTUNE)
    ds = ds.map(concat_target, tf.data.AUTOTUNE)

    if background.DEBUG:
        ds = ds.take(8)

    if cache:
        ds = ds.cache()

    samples_num = 0
    if calculate_sample_num:
        samples_num = ds.reduce(0, lambda x, _: x + 1).numpy()

    if shuffle:
        if shuffle == -1:
            ds = ds.shuffle(samples_num, reshuffle_each_iteration=True)
        else:
            ds = ds.shuffle(shuffle, reshuffle_each_iteration=True)

    if to_repeat:
        ds = ds.repeat()

    if batch_size:
        ds = ds.padded_batch(
            batch_size, padding_values=(background.PAD_x, background.PAD_y),
            padded_shapes=([background.X_max_len], [background.X_max_len, 2]), drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, samples_num

# define validation and training sets
val_len = 5
# val_len = 10
if background.DEBUG:
    val_len = 1

val_files = tffiles[:val_len]
if background.DEBUG:
    train_files = tffiles[val_len:val_len+1]
else:
    train_files = tffiles[val_len:]

# Shuffle dataset each time to get training and testing dataset
train_dataset, num_train = get_tfrec_dataset(train_files, shuffle = -1, batch_size = background.batch_size,
                                             cache = True, to_filter = 'filter_2', calculate_sample_num = True,
                                             to_repeat = True)

val_dataset, num_val = get_tfrec_dataset(val_files, shuffle = False, batch_size = background.val_batch_size,
                                         cache = True, to_filter = 'filter_1', calculate_sample_num = True)
print(num_train) # 306251
print(num_val) # 5512

batch = next(iter(val_dataset))
print(batch[0].shape, # [5512, 206]
      batch[1].shape) # [5512, 206, 2]
