# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:25:28 2017

@author: rcrescenzi
"""

import os

import numpy as np
import pandas as pd


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
prior = pd.read_csv(data_dir + "/order_products__prior.csv",
                    usecols=["order_id", "product_id"],
                    dtype={"order_id": np.uint32, "product_id": np.uint16})


prior = prior.join(orders[["user_id", "order_number"]], on="order_id")
prior = prior.join(prior.groupby("user_id").order_number.max().rename("max_order"), on="user_id")

evalset = prior.duplicated(["user_id", "product_id"])
evalset = np.invert(evalset)
evalset = prior[["user_id", "product_id"]][evalset]

trainset = prior[prior.order_number < prior.max_order].duplicated(["user_id", "product_id"])
trainset = np.invert(trainset)
trainset = prior[prior.order_number < prior.max_order][["user_id", "product_id"]][trainset]

prior = prior[prior.order_number == prior.max_order]
prior["reordered"] = 1
trainset = trainset.join(prior.set_index(["user_id", "product_id"]).reordered,
                         on=["user_id", "product_id"])
trainset.reordered.fillna(0, inplace=True)
trainset["reordered"] = trainset.reordered.astype(np.uint8)
del prior

real = pd.read_csv(data_dir + "/order_products__train.csv")
real = real.join(orders.user_id, on="order_id")
evalset = evalset.join(real.set_index(["user_id", "product_id"]).reordered,
                       on=["user_id", "product_id"])
evalset.reordered.fillna(0, inplace=True)
evalset["reordered"] = evalset.reordered.astype(np.uint8)

real = real[real.reordered == 1].groupby("user_id").product_id.apply(set)

orders = orders.join(orders.groupby("user_id").order_number.max().rename("max_order"), on="user_id")

evalorders = orders[orders.order_number == orders.max_order]
evalset = evalset.join(evalorders.drop(['eval_set', 'max_order'], axis=1).set_index('user_id'), on="user_id")

trainorders = orders[orders.order_number == (orders.max_order - 1)]
trainset = trainset.join(trainorders.drop(['eval_set', 'max_order'], axis=1).set_index('user_id'), on="user_id")
del evalorders, trainorders

train_users = orders[orders.eval_set == "train"].user_id
test_users = orders[orders.eval_set == "test"].user_id

trainset = trainset.fillna(-1)
trainset["days_since_prior_order"] = trainset.days_since_prior_order.astype(np.uint8)
evalset = evalset.fillna(-1)
evalset["days_since_prior_order"] = evalset.days_since_prior_order.astype(np.uint8)

real.to_pickle(os.path.join(target_dir, "eval_real.p"))
train_users.to_pickle(os.path.join(target_dir, "train_users.p"))
test_users.to_pickle(os.path.join(target_dir, "test_users.p"))
trainset.to_pickle(os.path.join(target_dir, "train.p"))
evalset.to_pickle(os.path.join(target_dir, "eval.p"))
