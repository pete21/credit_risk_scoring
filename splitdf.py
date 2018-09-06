# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
from collections import OrderedDict
from condition_fun import *


def split_df(dt, y=None, ratio=0.7, seed=186):
    '''
    Split a dataset
    ------
    Split a dataset into train and test
    
    Params
    ------
    dt: A data frame.
    y: Name of y variable, default is NULL. The input data will split 
        based on the predictor y, if it is provide.
    ratio: A numeric value, default is 0.7. It indicates the ratio of 
        total rows contained in one split, must less than 1.
    seed: A random seed, default is 186.
    
    Returns
    ------
    dict
        a dictionary of train and test
    
    '''
    
    dt = dt.copy(deep=True)
    # remove date/time col
    dt = rmcol_datetime_unique1(dt)
    # replace "" by NA
    dt = rep_blank_na(dt)
    
    # set ratio range
    if isinstance(ratio, float): ratio = [ratio]
    if not all(isinstance(i, float) for i in ratio) or len(ratio)>2 or sum(ratio)>1:
        warnings.warn("Incorrect inputs; ratio must be a numeric that length equal to 1 and less than 1. It was set to 0.7.")
        ratio = [0.7, 0.3]
    else:
        ratio_ = 1.0-sum(ratio)
        if (ratio_ > 0): ratio = ratio + [ratio_]
    # split into train and test
    if y is None:
        train = dt.sample(frac=ratio[0], random_state=seed).sort_index()
        test = dt.iloc[list(set(dt.index.tolist()).difference(set(train.index.tolist())))].sort_index()
    else:
        train = dt.groupby(y)\
          .apply(lambda x: x.sample(frac=ratio[0], random_state=seed))\
          .reset_index(level=y, drop=True)\
          .sort_index()
        test = dt.iloc[list(set(dt.index.tolist()).difference(set(train.index.tolist())))].sort_index()
        if len(ratio) == 3:
            test = test.groupby(y)\
                .apply(lambda x: x.sample(frac=ratio[1]/sum(ratio[1:]), random_state=seed))\
                .reset_index(level=y, drop=True)\
                .sort_index()
    # return
    rt = OrderedDict()
    rt['train'] = train
    rt['test'] = test
    return rt

