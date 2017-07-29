# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:54:56 2017

@author: Rafael
"""


import os

import numpy as np
import pandas as pd

from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score

for data_dir in ["C:\\Users\Rafael\\Documents\\data\\instacart\\raw",
                 "C:/Users/rcrescenzi/Documents/Personal/data/instacart/raw"]:
    if os.path.exists(data_dir):
        print(data_dir)
        break

target_dir = os.path.abspath(os.path.join(data_dir, "../mock_df"))

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

orders = pd.read_csv(data_dir + "/orders.csv", index_col="order_id",
                     dtype={
                        'order_id': np.uint32,
                        'user_id': np.uint32,
                        'eval_set': 'category',
                        'order_number': np.uint8,
                        'order_dow': np.uint8,
                        'order_hour_of_day': np.uint8,
                        'days_since_prior_order': np.float32})

valid_users = orders[orders.eval_set == "train"].user_id.sample(frac=0.1)
orders = orders[orders.user_id.isin(valid_users)]

def make_df(path):
    res = []
    for i, cand in enumerate(pd.read_csv(path, chunksize=100000)):
        temp =  cand.join(orders[["user_id", "order_number"]], on="order_id")
        temp = temp[temp.user_id.isin(valid_users)]
        res.append(temp)
        if (i % 100) == 0:
            print("haciendo", i+1)
            print(temp)
    return pd.concat(res)
data = make_df(data_dir + "/order_products__train.csv")
data = pd.concat([data, make_df(data_dir + "/order_products__prior.csv")])

em_pids = int(data.product_id.max() + 1)
max_order = int(data.order_number.max())
max_order_size = int(data.groupby("order_id").count().max().product_id)
EMBEDDING_SIZE = 10
BATCH_SIZE = 1

def pad(a):
    return np.pad(a, (max_order_size - a.shape[0], 0), "constant", constant_values=(0, 0))

data_orders = data.groupby(["user_id", "order_number"]).product_id.apply(lambda x: pad(x.values))
data_labels = data[data.reordered == 1].groupby(["user_id", "order_number"]).product_id.apply(lambda x: x.values)

temp = data.groupby("product_id").reordered.sum()
temp = temp[temp >= 1000].index
data = data[data.product_id.isin(temp)]
del temp
data["product_id"] = data.product_id.factorize()[0] + 1

pids = int(data.product_id.max() + 1)

data_labels4train = data.groupby(["user_id", "order_number"]).product_id.apply(lambda x: x.values)

data = pd.concat([data_orders.rename("compras"),
                  data_labels.rename("recompras"),
                  data_labels4train.rename("label")], axis=1)
del data_orders, data_labels

data = data.sort_index()
data["recompras"] = data.recompras.shift(-1)
data["label"] = data.label.shift(-1)

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed

model = Sequential()
model.add(TimeDistributed(Embedding(input_dim=em_pids, output_dim=EMBEDDING_SIZE, mask_zero=True), input_shape=(max_order, max_order_size)))
model.add(TimeDistributed(LSTM(20)))
model.add(LSTM(40))
model.add(Dense(80, activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dense(pids, activation="sigmoid"))
model.compile(optimizer='RMSprop', loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

def onehot(vec, pids=pids):
    if vec is np.nan:
        return np.zeros((pids,))
    else:
        return to_categorical(vec, pids).sum(axis=0)

def train_gen(data, last=False):
    users = data.index.levels[0].values
    zeros = np.zeros((max_order_size,))
    while True:
        for user in np.random.choice(users, len(users), replace=False):
            temp = data.xs(user)
            if not last:
                length = np.random.choice(temp.index[:-2], 1)[0]
                temp = temp.loc[:length]
            else:
                temp = temp.iloc[:-1]
            compras = temp.compras.values
            pad = int(max_order - temp.shape[0])
            X = []
            for i in range(max_order):
                if i < pad:
                    X.append(zeros)
                else:
                    X.append(compras[pad - i])
            yield np.asarray([np.vstack(X)]), np.asarray([onehot(temp.label.iloc[-1])])

tgen = train_gen(data)
test_gen = train_gen(data, last=True)

steps = data.index.levels[0].shape[0]
steps = int(steps / 100)

a = [next(test_gen) for _ in range(steps)]
a0 = [t[0] for t in a]
a1 = [t[1] for t in a]
del a
a0 = np.concatenate(a0)
a1 = np.concatenate(a1)

from keras.callbacks import Callback

class CustomCallback(Callback):

    def on_epoch_end(self, epoch, log):
        preds = self.model.predict(self.validation_data[0])
        if np.allclose(preds[0], preds[1]):
            print("all close")
            return 0
        preds = np.argsort(-preds, axis=1)
        res = []
        for pred, real in zip(preds, self.validation_data[1]):
            real = np.where(real == 1)[0]
            if real.shape[0] == 0:
                res.append(0.5)
            else:
                pred = pred[:real.shape[0]]
                res.append(f1_score(real, pred, average='micro'))
        print("F1:", np.mean(res))

callback = CustomCallback()
model.fit_generator(tgen, steps_per_epoch=steps, epochs=10,
                    validation_data=(a0, a1), verbose=2,
                    callbacks=[callback])
