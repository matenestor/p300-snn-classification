#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from math import factorial

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import metrics
from sklearn.model_selection import train_test_split, ShuffleSplit
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import AveragePooling1D, BatchNormalization,                                     Conv1D, Dense, Dropout, Flatten, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import nengo
import nengo_dl
from nengo.utils.filter_design import cont2discrete


# In[35]:


# amount of cross-validation splits
SPLITS = 5
# size of train-test split
SPLIT_SIZE = 0.8
BATCH = 50
EPOCHS = 30

# True -- average random brainwave signal samples among people
# False -- average single brainwave signal of one person (smoothening)
AVERAGE_SAMPLES = True
# when AVERAGE_SAMPLES is True, it is amount of samples
#   from people, that will be averaged together
# when AVERAGE_SAMPLES is False, it is a sliding window size
#   used to smoothen one signal from one person
AVERAGING_AMOUNT = [3, 6, 9, 12, 15]
# useful only when AVERAGE_SAMPLES is True
TRAIN_DATA_AMOUNT = 10000

# set seed to ensure this experiment is reproducible
SEED = 0
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# splitter for cross-validation
shs = ShuffleSplit(n_splits=SPLITS, train_size=SPLIT_SIZE, random_state=SEED)


# In[3]:


def export_data(results_final, filename='results'):
    df = pd.DataFrame(results_final)
    df.to_csv(f'{filename}.csv', index=False)
    df.to_excel(f'{filename}.xlsx', index=False)


# In[4]:


def get_results_final_dict():
    return {
        'Model': [],
        'Averaging': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-score': [],
        'ConfM-00': [],
        'ConfM-01': [],
        'ConfM-10': [],
        'ConfM-11': [],
    }


# # Data

# ### Load and filter

# In[5]:


# load matrix with P300 dataset
mat = loadmat('VarekaGTNEpochs.mat')
target_data, non_target_data = mat['allTargetData'], mat['allNonTargetData']

# filtering needs to be done on both arrays separately,
# because of averging later, but it is done only once in total

# filter noise above 100 uV
threshold = 100.0
filter_target_data, filter_non_target_data = [], []
# filter target data
for i in range(target_data.shape[0]):
    if np.max(np.abs(target_data[i])) <= threshold:
        filter_target_data.append(target_data[i])
# filter non-target data
for i in range(non_target_data.shape[0]):
    if np.max(np.abs(non_target_data[i])) <= threshold:
        filter_non_target_data.append(non_target_data[i])

# replace loaded data with filtered data
target_data = np.array(filter_target_data)
non_target_data = np.array(filter_non_target_data)

print('Target data size:', target_data.shape)
print('Non-target data size:', non_target_data.shape)


# ### Averaging

# In[34]:


def get_averages_over_signal(data, window_size):
    # NOTE:
    # It would be possible to speed up this function with
    # `np.average(sliding_window_view(data, window_size, axis=2), axis=-1)`
    # but `numpy.lib.stride_tricks.sliding_window_view` is available since NumPy 1.20
    # and NumPy 1.20 does not work with Tensorflow 2.5.0.
    # Specifically tensor operations in NumPy with LSTM layers.
    # `numpy.lib.stride_tricks.as_strided` is not memory safe and could damage data.
    if window_size < 2:
        return data

    averages = []

    # 1. samples
    for i in range(data.shape[0]):
        averaged.append([[],[],[]])
        # 2. channels
        for j in range(data.shape[1]):
            # 3. features -- floating window
            for k in range(data.shape[2] - window_size + 1):
                averages[i][j].append(np.average(data[i][j][k:k+window_size]))

    return np.array(averages)


# In[7]:


# divided by 2, because this function is called twice,
# for target and for non-target data
def get_averages_over_samples(data, amount_to_average, amount_new_data=TRAIN_DATA_AMOUNT//2):
    if amount_to_average < 2:
        return data

    # get `amount_new_data` amount of vectors
    # with length `amount_to_average`
    random_choices = np.random.choice(
        np.arange(data.shape[0]),
        (amount_new_data, amount_to_average)
    )

    averaged = []
    # use vectors with random numbers to choose samples to average
    for rchoice in random_choices:
        tmp = []
        for i in rchoice:
            tmp.append(data[i])
        averaged.append(np.average(tmp, axis=0))

    return np.array(averaged)


# ### Labeling

# In[8]:


def get_data_labels(targets, non_targets):
    # target numbers are labeled with vector [1, 0]
    target_labels = np.tile(np.array([1, 0]), (targets.shape[0], 1))
    # non-target numbers are labeled with vector [1, 0]
    non_target_labels = np.tile(np.array([0, 1]), (non_targets.shape[0], 1))

    # concatenate target and non-target sampels and labels
    samples = np.concatenate((targets, non_targets))
    labels = np.vstack((target_labels, non_target_labels))

    # reshape to single vector and for correct inputs
    samples = samples.reshape((samples.shape[0], 1, -1))

    return samples, labels


# ### Train/test split

# In[9]:


# create train and test sets
def get_train_test_data(samples, labels):
    x, x_test, y, y_test = train_test_split(
        samples, labels, train_size=SPLIT_SIZE, random_state=SEED)
    return x, x_test, y, y_test


# ### Metrics

# In[10]:


def get_metrics(predictions, y_test):
    # normalize predictions and reality vectors
    preds = np.argmax(predictions, axis=-1)
    reality = np.argmax(y_test, axis=-1)[:predictions.shape[0]]
    # calculate metrics
    accuracy = metrics.accuracy_score(y_true=reality, y_pred=preds)
    precision = metrics.precision_score(y_true=reality, y_pred=preds)
    recall = metrics.recall_score(y_true=reality, y_pred=preds)
    f1 = metrics.f1_score(y_true=reality, y_pred=preds)
    confusion_matrix = metrics.confusion_matrix(y_true=reality, y_pred=preds)

    return [accuracy, precision, recall, f1, confusion_matrix]


# # Models

# ### ANN Convolutional (model 1)

# In[11]:


def get_model_conv(inp_data):
    model = Sequential([
        Conv1D(filters=32, kernel_size=8, strides=4,
               padding='same', activation=relu,
               input_shape=(inp_data.shape[1], inp_data.shape[2])),
        BatchNormalization(),
        Dropout(rate=0.3, seed=SEED),
        AveragePooling1D(pool_size=4, strides=1, padding='same'),
        Flatten(),
        Dense(64, activation=relu),
        BatchNormalization(),
        Dropout(rate=0.4, seed=SEED),
        Dense(2, activation=softmax),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=['acc', 'mae',
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.Precision()]
    )
    return model


# ### ANN Long Short-Term Memory (model 2)

# In[12]:


def get_model_lstm():
    # NOTE: LSTM does not produce reproducible results
    model = Sequential([
        LSTM(units=32, activation=relu, return_sequences=True),
        BatchNormalization(),
        Dropout(rate=0.3, seed=SEED),
        LSTM(units=32, activation=relu),
        Dense(64, activation=relu),
        BatchNormalization(),
        Dropout(rate=0.4, seed=SEED),
        Dense(2, activation=softmax),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=['acc', 'mae',
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.Precision()]
    )
    return model


# ### Conversion of model 1 to spiking model 3

# In[13]:


def convert_ann2snn(model):
    converter = nengo_dl.Converter(
        model=model,
        swap_activations={
            tf.keras.activations.relu: nengo.SpikingRectifiedLinear()
        },
        scale_firing_rates=3000,
        synapse=0.1,
    )
    return converter


# ### LMU cell & SNN LMU (model 4)

# In[14]:


class LMUCell(nengo.Network):
    """Spiking version of LMU cell.
    source: https://www.nengo.ai/nengo-dl/v3.3.0/examples/lmu.html (06, 2021)
    """
    def __init__(self, units, order, theta, input_d, **kwargs):
        super().__init__(**kwargs)

        # compute the A and B matrices according to the LMU's mathematical derivation
        # (see the paper for details)
        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)

        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))

        A, B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")

        with self:
            nengo_dl.configure_settings(trainable=None)

            # create objects corresponding to the x/u/m/h variables in the above diagram
            self.x = nengo.Node(size_in=input_d)
            self.u = nengo.Node(size_in=1)
            self.m = nengo.Node(size_in=order)
            self.h = nengo_dl.TensorNode(tf.nn.tanh, shape_in=(units,), pass_time=False)

            # compute u_t from the above diagram.
            # note that setting synapse=0 (versus synapse=None) adds a one-timestep
            # delay, so we can think of any connections with synapse=0 as representing
            # value_{t-1}
            nengo.Connection(
                self.x, self.u, transform=np.ones((1, input_d)), synapse=None)
            nengo.Connection(self.h, self.u, transform=np.zeros((1, units)), synapse=0)
            nengo.Connection(self.m, self.u, transform=np.zeros((1, order)), synapse=0)

            # compute m_t
            # in this implementation we'll make A and B non-trainable, but they
            # could also be optimized in the same way as the other parameters
            conn_A = nengo.Connection(self.m, self.m, transform=A, synapse=0)
            self.config[conn_A].trainable = False
            conn_B = nengo.Connection(self.u, self.m, transform=B, synapse=None)
            self.config[conn_B].trainable = False

            # compute h_t
            nengo.Connection(
                self.x, self.h, transform=np.zeros((units, input_d)), synapse=None
            )
            nengo.Connection(
                self.h, self.h, transform=np.zeros((units, units)), synapse=0)
            nengo.Connection(
                self.m,
                self.h,
                transform=nengo_dl.dists.Glorot(distribution="normal"),
                synapse=None,
            )


# In[27]:


def get_model_lmu(inp_data):
    with nengo.Network(seed=SEED) as net:
        # input node for data
        inp = nengo.Node(np.ones(inp_data.shape[-1]))
        # LMU cell
        lmu1 = LMUCell(units=212, order=256,
                      theta=inp_data.shape[-1], input_d=inp_data.shape[-1])
        lmu2 = LMUCell(units=212, order=256,
                      theta=inp_data.shape[-1], input_d=212)
        # output node for probing result data
        out = nengo.Node(size_in=2)

        # input node is connected with LMU's `x` variable,
        # where input vectors flow into
        nengo.Connection(inp, lmu1.x, synapse=None)
        # LMU's hidden state is kept in a variable `h`
        # it is also an output connected to output node
        nengo.Connection(lmu1.h, lmu2.x,
                         transform=nengo_dl.dists.Glorot(), synapse=None)
        nengo.Connection(lmu2.h, out,
                         transform=nengo_dl.dists.Glorot(), synapse=None)

        # probe for collecting data
        p = nengo.Probe(target=out)

    return net


# # Simulation

# In[16]:


def fit_ann(model, x_train, y_train, x_val, y_val):
    model.fit(
        x=x_train, y=y_train, batch_size=BATCH, epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[EarlyStopping(patience=5, verbose=1, restore_best_weights=True)],
        verbose=1
    )
    return model


# In[17]:


def predict_ann(model, x_test, y_test):
    predictions = model.predict(x_test, batch_size=BATCH)
    return get_metrics(predictions, y_test)


# In[18]:


def predict_snn(network, x_test, y_test):
    # NOTE: SNN LMU is being predicted in `run_lmu` function immediatelly,
    #       because the trained simulator is closed there, so training is lost
    with nengo_dl.Simulator(network=network, minibatch_size=BATCH) as sim:
        predictions = sim.predict(x_test)
    # retrieve predictions matrix from Nengo object
    predictions = np.array(list(predictions.values())[0])
    return get_metrics(predictions, y_test)


# In[19]:


def run_lmu(network, x_train, y_train, x_val, y_val, x_test, y_test):
    with nengo_dl.Simulator(network=network, minibatch_size=BATCH) as sim:
        sim.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['acc', 'mae'],
        )
        # matrices with labels for training and validation
        # need to be reshaped to 3 dimensions, in order to fit the network
        sim.fit(
            x=x_train,
            y=y_train.reshape((y_train.shape[0], 1, -1)),
            validation_data=(x_val, y_val.reshape((y_val.shape[0], 1, -1))),
            epochs=EPOCHS,
            callbacks=[EarlyStopping(patience=5, verbose=1, restore_best_weights=True)],
            verbose=1
        )

        # test SNN network
        predictions = sim.predict(x_test)
        # retrieve predictions matrix from Nengo object
        predictions = np.array(list(predictions.values())[0])

    return get_metrics(predictions, y_test)


# In[33]:


def run_simulation(x, x_test, y, y_test):
    iteration = 1
    models_metrics = {
        'ann_conv': [],
        'ann_lstm': [],
        'snn_conv': [],
        'snn_lmu': [],
    }

    for train_idx, val_idx in shs.split(x):
        print(f'--- iteration: {iteration}/{SPLITS} ---')
        iteration += 1

        # split data to train and validation set
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        # train ANN models
        print('--- ANN conv')
        ann_conv = fit_ann(get_model_conv(x), x_train, y_train, x_val, y_val)
        print('--- ANN lstm')
        ann_lstm = fit_ann(get_model_lstm(), x_train, y_train, x_val, y_val)
        # convert trained ann_conv to spiking convolutional network
        print('--- SNN conv')
        snn_conv = convert_ann2snn(ann_conv)
        # train SNN model and immediately test
        print('--- SNN lmu')
        snn_lmu_results = run_lmu(
            get_model_lmu(x), x_train, y_train, x_val, y_val, x_test, y_test
        )

        # test NN models
        models_metrics['ann_conv'].append(predict_ann(ann_conv, x_test, y_test))
        models_metrics['ann_lstm'].append(predict_ann(ann_lstm, x_test, y_test))
        models_metrics['snn_conv'].append(predict_snn(snn_conv.net, x_test, y_test))
        models_metrics['snn_lmu'].append(snn_lmu_results)

        # reshape confusion matrices to 1D vector for easier averaging
        models_metrics['ann_conv'][-1][-1] = np.squeeze(
            models_metrics['ann_conv'][-1][-1].reshape(1, -1))
        models_metrics['ann_lstm'][-1][-1] = np.squeeze(
            models_metrics['ann_lstm'][-1][-1].reshape(1, -1))
        models_metrics['snn_conv'][-1][-1] = np.squeeze(
            models_metrics['snn_conv'][-1][-1].reshape(1, -1))
        models_metrics['snn_lmu'][-1][-1] = np.squeeze(
            models_metrics['snn_lmu'][-1][-1].reshape(1, -1))

    # average all results from cross-validation
    models_metrics['ann_conv'] = np.average(models_metrics['ann_conv'], axis=0)
    models_metrics['ann_lstm'] = np.average(models_metrics['ann_lstm'], axis=0)
    models_metrics['snn_conv'] = np.average(models_metrics['snn_conv'], axis=0)
    models_metrics['snn_lmu'] = np.average(models_metrics['snn_lmu'], axis=0)

    return models_metrics


# In[32]:


results_final = get_results_final_dict()

for avg in AVERAGING_AMOUNT:
    if AVERAGE_SAMPLES and avg > 1:
        # averages from multiple samples
        target_data_averaged = get_averages_over_samples(target_data, avg)
        non_target_data_averaged = get_averages_over_samples(non_target_data, avg)
    else:
        print(f'> Averaging signals with amount {avg}')
        # averages (smoothened) brain signals
        target_data_averaged = get_averages_over_signal(target_data, avg)
        non_target_data_averaged = get_averages_over_signal(non_target_data, avg)

    # create samples and their labels from averaged data
    samples, labels = get_data_labels(target_data_averaged, non_target_data_averaged)

    if AVERAGE_SAMPLES and avg <= 1:
        print('> Averaging amount is smaller than 2. '
              'Proceeding with standard train-test data split')

    # create train and test datasets
    if AVERAGE_SAMPLES and avg > 1:
        # when samples are averaged, the original data is used as a test dataset
        samples_test, labels_test = get_data_labels(target_data, non_target_data)
        x, x_test, y, y_test = samples, samples_test, labels, labels_test
    else:
        x, x_test, y, y_test = get_train_test_data(samples, labels)

    print(f'----- AVERAGING AMOUNT: {avg} -----')
    results = run_simulation(x, x_test, y, y_test)

    # cache results from current window size run for later export
    for model_name, res in results.items():
        results_final['Model'].append(model_name)
        results_final['Averaging'].append(avg)
        results_final['Accuracy'].append(res[0])
        results_final['Precision'].append(res[1])
        results_final['Recall'].append(res[2])
        results_final['F1-score'].append(res[3])
        results_final['ConfM-00'].append(res[4][0])
        results_final['ConfM-01'].append(res[4][1])
        results_final['ConfM-10'].append(res[4][2])
        results_final['ConfM-11'].append(res[4][3])

export_data(results_final)

print('--- DONE sample averaging ---')

