# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:26:51 2018

@author: rcrescenzi
"""

import numpy as np
import pandas as pd

from keras.layers import Layer, Dense
from keras import initializers
from keras import backend as K
from keras.models import Sequential

import seaborn as sns

class RBFLayer(Layer):
    def __init__(self, output_dim, normalized=True, **kwargs):
        self.output_dim = output_dim
        self.normalized = normalized
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RBFLayer, self).build(input_shape)
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=initializers.RandomUniform(0.0, 1.0),
                                       trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim, ),
                                    initializer=initializers.constant(1.0),
                                    trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=initializers.constant(1.0),
                                     trainable=True)

    def call(self, inputs, **kwargs):
        c = K.expand_dims(self.centers, axis=1)
        h = K.transpose(c - inputs)
        res = K.exp(self.bias - self.betas * K.sum(h ** 2, axis=0))
        if self.normalized:
            return res / K.expand_dims(K.sum(res, axis=1), axis=1)
        else:
            return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim



X_train = np.random.rand(10000, 2)
y_train = (np.sum(np.abs(X_train - [0.5, 0.5]), axis=1) < 0.5).astype(int)
data = pd.DataFrame(X_train, columns=["x", "y"])
data["target"] = y_train
sns.pairplot(x_vars=["x"], y_vars=["y"], data=data, hue="target", size=8)

radial_model = Sequential([
    RBFLayer(5, input_shape=(2,)),
    Dense(1, activation="sigmoid")
])

radial_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

print("*" * 10, " RADIAL -- CUADRADO")
radial_model.fit(X_train, y_train, epochs=25)
data["pred_radial"] = radial_model.predict(X_train).round(1)
sns.pairplot(x_vars=["x"], y_vars=["y"], data=data, hue="pred_radial", size=8)


MLP_model = Sequential([
    Dense(5, input_shape=(2,), activation="relu"),
    Dense(1, activation="sigmoid")
])

MLP_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

print("*" * 10, " MLP 1-- CUADRADO")
MLP_model.fit(X_train, y_train, epochs=25)
data["pred_MLP"] = MLP_model.predict(X_train).round(1)
sns.pairplot(x_vars=["x"], y_vars=["y"], data=data, hue="pred_MLP", size=8)



MLP_model = Sequential([
    Dense(10, input_shape=(2,), activation="relu"),
    Dense(5, activation="relu"),
    Dense(1, activation="sigmoid")
])

MLP_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

print("*" * 10, " MLP 2-- CUADRADO")
MLP_model.fit(X_train, y_train, epochs=25)
data["pred_MLP"] = MLP_model.predict(X_train).round(1)
sns.pairplot(x_vars=["x"], y_vars=["y"], data=data, hue="pred_MLP", size=8)






#############


X_train = np.random.rand(10000, 2)
y_train = (np.power(np.sum(np.power(X_train - [0.5, 0.5], 2), axis=1), 0.5) < 0.5).astype(int)
data = pd.DataFrame(X_train, columns=["x", "y"])
data["target"] = y_train
sns.pairplot(x_vars=["x"], y_vars=["y"], data=data, hue="target", size=8)

radial_model = Sequential([
    RBFLayer(5, input_shape=(2,)),
    Dense(1, activation="sigmoid")
])

radial_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

print("*" * 10, " RADIAL -- REDONDO")
radial_model.fit(X_train, y_train, epochs=50)
data["pred_radial"] = radial_model.predict(X_train).round(1)
sns.pairplot(x_vars=["x"], y_vars=["y"], data=data, hue="pred_radial", size=8)


MLP_model = Sequential([
    Dense(10, input_shape=(2,), activation="relu"),
    Dense(5, activation="relu"),
    Dense(1, activation="sigmoid")
])

MLP_model.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

print("*" * 10, " MLP 2 -- REDONDO")
MLP_model.fit(X_train, y_train, epochs=50)
data["pred_MLP"] = MLP_model.predict(X_train).round(1)
sns.pairplot(x_vars=["x"], y_vars=["y"], data=data, hue="pred_MLP", size=8)

from keras.layers import Input, concatenate
from keras.models import Model

input1 = Input(shape=(2, ))
dense1 = RBFLayer(5)(input1)
input2 = Input(shape=(2, ))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
merged = concatenate([dense1, dense2])
out = Dense(1, activation='sigmoid', name='output_layer')(merged)
ensamble = Model(inputs = [input1, input2], outputs = [out])
ensamble.compile(optimizer='rmsprop',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
print("*" * 10, " ENSAMBLE -- REDONDO")
ensamble.fit([X_train, X_train], y_train, epochs=50)
data["pred_ensamble"] = ensamble.predict([X_train, X_train]).round(1)
sns.pairplot(x_vars=["x"], y_vars=["y"], data=data, hue="pred_ensamble", size=8)
ensamble.layers[-1].get_weights()


X_train = np.random.rand(10000, 2)
y_train = np.arctan2(X_train[:, 0] - 0.5, X_train[:, 1] - 0.5)
data = pd.DataFrame(X_train, columns=["x", "y"])
data["target"] = y_train.round(1)
sns.pairplot(x_vars=["x"], y_vars=["y"], data=data, hue="target", size=8)

