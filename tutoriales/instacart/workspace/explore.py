# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 20:20:53 2017

@author: Rafael
"""
import os

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

for data_dir in ["C:\\Users\Rafael\\Documents\\data\\instacart\\mock_df",
                 "C:/Users/rcrescenzi/Documents/Personal/data/instacart/mock_df"]:
    if os.path.exists(data_dir):
        print(data_dir)
        break

real = pd.read_pickle(os.path.join(data_dir, "eval_real.p"))
train_users = pd.read_pickle(os.path.join(data_dir, "train_users.p"))
test_users = pd.read_pickle(os.path.join(data_dir, "test_users.p"))
trainset = pd.read_pickle(os.path.join(data_dir, "train.p"))
evalset = pd.read_pickle(os.path.join(data_dir, "eval.p"))

real = real.loc[train_users]
evalset = evalset[evalset.user_id.isin(train_users)]

mds = [3, 5, 8, 10]
eval_ps = [1/i for i in range(1, 20)]
res = pd.DataFrame([], index=mds, columns=eval_ps)

for md in mds:
    learner = LGBMClassifier(n_estimators=10000, max_depth=md)

    learner.fit(trainset.drop("reordered", axis=1), trainset.reordered,
                eval_metric="auc", early_stopping_rounds=10,
                eval_set=[(trainset.drop("reordered", axis=1), trainset.reordered),
                          (evalset.drop("reordered", axis=1), evalset.reordered)])

    preds = learner.predict_proba(evalset.drop("reordered", axis=1))[:, -1]

    for p in eval_ps:
        ppreds = evalset[preds > p]
        ppreds = ppreds.groupby("user_id").product_id.apply(set)
        ppreds.name = "preds"
        real.name = "real"

        comp = pd.concat([real, ppreds], axis=1)
        temp = pd.Series([set([0])] * comp.shape[0], index = comp.index)
        comp.real.fillna(temp, inplace=True)
        comp.preds.fillna(temp, inplace=True)
        comp["tp"] = comp.apply(lambda x: len(x["real"].intersection(x.preds)), axis=1)
        comp["acc"] = comp.tp / comp["preds"].apply(len)
        comp["recall"] = comp.tp / comp["real"].apply(len)
        comp["f1"] = 2 * (comp["acc"] * comp["recall"]) / (comp["acc"] + comp["recall"])
        comp.f1.fillna(0, inplace=True)
        res.loc[md, p] = comp.f1.mean()
        print(res)