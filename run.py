# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from splitdf import split_df
from info_value import iv
# from .info_ent_indx_gini import (ig, ie)
from var_filter import var_filter
from woebin import (woebin, woebin_ply, woebin_plot, woebin_adj)
from perf import (perf_eva, perf_psi)
from scorecard import (scorecard, scorecard_ply)
import pandas as pd

# Logistic Regression

dat = pd.read_csv('data/creditdataset.csv')

# filter variable via missing rate, iv, identical value rate
dt_s = var_filter(dat, y="creditability", iv_limit=0.02)

# breaking dt into train and test
train, test = split_df(dt_s, 'creditability', ratio = [0.7, 0.3]).values()

# woe binning ------

bins = woebin(dt_s, y="creditability")
woebin_plot(bins)

# binning adjustment
# # adjust breaks interactively
# breaks_adj = woebin_adj(dt_s, "creditability", bins) 
# # or specify breaks manually
breaks_adj = {
    'age.in.years': [27, 35, 45, 55],
    'other.debtors.or.guarantors': ["none", "co-applicant%,%guarantor"]
}
bins_adj = woebin(dt_s, y="creditability", breaks_list=breaks_adj)


ivlist=iv(dt_s, y="creditability")
ivlist


# converting train and test into woe values
train_woe = woebin_ply(train, bins_adj)
test_woe = woebin_ply(test, bins_adj)

y_train = train_woe.loc[:,'creditability']
X_train = train_woe.loc[:,train_woe.columns != 'creditability']
y_test = test_woe.loc[:,'creditability']
X_test = test_woe.loc[:,train_woe.columns != 'creditability']

# logistic regression ------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', n_jobs=-1)
lr.fit(X_train, y_train)
lr.coef_
lr.intercept_

# predicted proability
train_pred = lr.predict_proba(X_train)[:,1]
test_pred = lr.predict_proba(X_test)[:,1]

# performance ks & roc ------
train_perf = perf_eva(y_train, train_pred, title = "train")
test_perf = perf_eva(y_test, test_pred, title = "test")

#perf_eva(y_test, test_pred, plot_type = ["ks","lift","roc","pr"])


# score ------
card = scorecard(bins_adj, lr, X_train.columns)

# credit score
train_score = scorecard_ply(train, card, print_step=0)
test_score = scorecard_ply(test, card, print_step=0)


# psi
perf_psi(
  score = {'train':train_score, 'test':test_score},
  label = {'train':y_train, 'test':y_test}
)
