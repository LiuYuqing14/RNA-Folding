import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil
import math
import pandas as pd
import gc
import os

# TPU setup
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local") # "local" for 1VM TPU
    strategy = tf.distribute.TPUStrategy(tpu)
    print("on TPU")
    print("REPLICAS: ", strategy.num_replicas_in_sync)
except:
    strategy = tf.distribute.get_strategy()

# Configurations
DEBUG = False

PAD_x = 0.0
PAD_y = np.nan
X_max_len = 206


batch_size = 128
# batch_size = 512
val_batch_size = 5512

if DEBUG:
    batch_size = 2 # define a smaller batch size for higher accuracy
    # batch_size = 4
    # batch_size = 8
    val_batch_size = 2 # validation split
    # val_batch_size = 4
    # val_batch_size = 8
num_vocab = 5
hidden_dim = 192
