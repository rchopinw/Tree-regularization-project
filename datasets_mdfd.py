# Generating practical datasets
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from numba import cuda
import _pickle as cPickle

files = os.listdir('D:\\tree regularization project\\Beat')

dats = []
for i in files:
    dats.append(np.array(pd.read_csv('./Beat/'+i, engine='python', encoding='utf_8_sig')))
label = []
data_raw = []
for i, j in enumerate(dats):
    dum = [0, 0, 0, 0]
    dum[i] = 1
    label += [dum for _ in range(j.shape[0])]
    data_raw += j.tolist()
label = np.array(label)
data_raw = np.array(data_raw)
# approach 1
x_train, x_test, y_train, y_test = train_test_split(data_raw, label, test_size=0.25, stratify=label)
x_train = np.reshape(np.array(x_train), (-1, 1)).T
x_test = np.reshape(np.array(x_test), (-1, 1)).T
y_tr = []
y_ts = []
for i in y_train:
    y_tr += [i for _ in range(250)]
for i in y_test:
    y_ts += [i for _ in range(250)]
y_train = np.array(y_tr).T
y_test = np.array(y_ts).T
fcpt_train = np.arange(0, len(x_train.T) + 250, 250)
fcpt_test = np.arange(0, len(x_test.T) + 250, 250)
if not os.path.isdir('./data'):
    os.mkdir('./data')

with open('./data/train.pkl', 'wb') as fp:
    cPickle.dump({'X': x_train, 'F': fcpt_train, 'y': y_train}, fp)

with open('./data/test.pkl', 'wb') as fp:
    cPickle.dump({'X': x_test, 'F': fcpt_test, 'y': y_test}, fp)

# approach 2
x_train, x_test, y_train, y_test = train_test_split(data_raw, label, test_size=0.25, stratify=label)
x_train = list(map(lambda x: [x[i*5:(i+1)*5] for i in range(50)], x_train.tolist()))
x_train = np.reshape(np.array(x_train), (-1, 5)).T
x_test = list(map(lambda x: [x[i*5:(i+1)*5] for i in range(50)], x_test.tolist()))
x_test = np.reshape(np.array(x_test), (-1, 5)).T
y_tr = []
y_ts = []
for i in y_train:
    y_tr += [i for _ in range(50)]
for i in y_test:
    y_ts += [i for _ in range(50)]
y_train = np.array(y_tr).T
y_test = np.array(y_ts).T
fcpt_train = np.arange(0, len(x_train.T) + 50, 50)
fcpt_test = np.arange(0, len(x_test.T) + 50, 50)
if not os.path.isdir('./data'):
    os.mkdir('./data')

with open('./data/train1.pkl', 'wb') as fp:
    cPickle.dump({'X': x_train, 'F': fcpt_train, 'y': y_train}, fp)

with open('./data/test1.pkl', 'wb') as fp:
    cPickle.dump({'X': x_test, 'F': fcpt_test, 'y': y_test}, fp)


